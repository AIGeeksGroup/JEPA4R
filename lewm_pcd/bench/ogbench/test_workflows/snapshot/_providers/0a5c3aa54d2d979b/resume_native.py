"""Resume only the unchanged Native stages after the handoff metadata failure."""
import argparse
import importlib.util
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('original_relay', ROOT / 'relay_v1/relay_v2.py')
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def preflight():
    assert relay.sha(ROOT / 'relay_v1/relay_v2.py') == '01630d69c8210d32de9b9ae3a721bd69c8a9c650c6001de761f11c52c776ec48'
    assert (ROOT / 'relay_v1/runtime/actual_exit_code').read_text().strip() == '1'
    old = relay.load(ROOT / 'relay_v1/runtime/status.json')
    assert old['stage'] == 'failed' and "KeyError: 'goal_dims'" in old['error']
    path = ROOT / 'relay_v1/formal_plan_v2.json'
    assert relay.sha(path) == 'bd76211d4db28d96f811aece34ff3c66723d76467f56351a209bae9cae2f8927'
    plan = relay.load(path)
    pointer = relay.native_ready_pointer(plan)
    assert relay.sha(pointer['path']) == pointer['sha256']
    handoff = relay.load(pointer['path'])
    assert handoff['ready'] and set(handoff['modes']) == {'M0', 'M6', 'M5'}
    assert handoff['selected_updates'] == 1000000 and handoff['goal_dims'] == 195
    for path, expected in handoff['source_sha256'].items():
        assert relay.sha(path) == expected
    for mode in ['M0', 'M6', 'M5']:
        receipt = relay.load(ROOT / f'relay_v1/runtime/{mode}_acceptance_receipt.json')
        assert receipt['status'] == 'accepted_fresh1M_fullstate_and_action' and receipt['exit_code'] == 0
        assert handoff['modes'][mode]['current_dims'] == plan['modes'][mode]['current_dims']
    return pointer, handoff


def main(check_only):
    pointer, handoff = preflight()
    if check_only:
        (HERE / 'check_only_receipt.json').write_text(json.dumps(dict(exit_code=0, handoff=pointer, modes=['M0', 'M6', 'M5'], training_or_acceptance_restarted=False), indent=2))
        return
    relay.RUN = HERE / 'runtime'
    relay.RUN.mkdir(exist_ok=False)
    code = 1
    try:
        for mode in ['M0', 'M6', 'M5']:
            for stage in handoff['modes'][mode]['stages']:
                assert not stage.get('cpu', False)
                relay.empty_GPU()
                relay.run(mode + '_native_' + stage['name'], stage['argv'])
                assert all(Path(path).read_text().strip() == '0' for path in stage['required_zero_exits'])
                proof = relay.load(stage['report'])
                assert proof['status'] == stage['status'] and proof['exit_code'] == 0
            final = relay.load(handoff['modes'][mode]['final_report'])
            assert final['exit_code'] == 0 and final['episodes'] == 250 and final['status'].startswith('passed')
        relay.status('all_three_native250_and_CPU_audits_complete', exit_code=0, training_restarted=False)
        code = 0
    except BaseException:
        relay.status('failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        (relay.RUN / 'actual_exit_code').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args().check_only)
