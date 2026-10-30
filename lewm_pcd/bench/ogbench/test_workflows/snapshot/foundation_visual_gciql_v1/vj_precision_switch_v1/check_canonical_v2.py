"""One canonical-N1 candidate only; unchanged old tolerance and paused-process guard."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    for key in ('source-spec','canonical-spec','input','protected-cache','output'):p.add_argument('--'+key,type=Path)
    p.add_argument('--paused-pids');args=p.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True,NN_forward=False,candidate='FP16 canonical1',warmups=2,trials=5)));return
    assert all(getattr(args,key) is not None for key in ('source_spec','canonical_spec','input','protected_cache','output','paused_pids'))
    here=Path(__file__).resolve().parent;diag=here.parent/'vj_official_forward_diagnostic_v1'
    import hashlib
    assert hashlib.sha256((diag/'benchmark.py').read_bytes()).hexdigest()=='3cb9a822987a2ad69127de49c3f5d1a9642b78fd86ab33accb90169e8ec0489e'
    u=module(diag/'benchmark.py','_frozen_diagnostic_guards');bridge=module(here/'precision_bridge_v2.py','_canonical_v2_bridge')
    emit=module(here/'emit_spec_v2.py','_canonical_v2_spec')
    import numpy as np
    import torch
    args.output.mkdir(parents=True,exist_ok=False);code=1;baseline=None;pids=[]
    try:
        pids=[int(pid) for pid in args.paused_pids.split(',')];assert len(pids)==len(set(pids))==11
        identities=[u.proc_identity(pid) for pid in pids];u.safety_check(pids,identities)
        baseline=u.cache_stats(args.protected_cache);assert baseline['episode_receipts']==baseline['latent_files']==59
        assert u.sha(args.source_spec)==u.SPEC_SHA
        assert u.read(args.canonical_spec)==emit.make_spec(u.read(args.source_spec))
        assert u.sha(args.input)=='429d502c0c7dd5ed5f08152efd7c35fb031689acaa5606bcb98a493f71e164c3'
        pixels=np.load(args.input,allow_pickle=False);u.validate_pixels(pixels,'BHWC')
        raw=torch.from_numpy(np.ascontiguousarray(pixels)).permute(0,3,1,2).contiguous();torch.set_num_threads(1);torch.manual_seed(0)
        u.save(args.output/'paused_baseline.json',dict(processes=identities,cache=baseline))
        # Both are the exact legacy FP32 method on one default-off model instance.
        fp32=bridge.load_encoder(args.source_spec,device='cuda');before32=u.fingerprint(fp32)
        legacy=module(bridge.v1.BASE,'_unchanged_legacy_adapter')
        with torch.inference_mode():
            legacy_value=legacy.FrozenImageEncoder.encode_uint8(fp32,raw)
            default_value=fp32.encode_uint8(raw)
            default_metrics=u.metrics(torch,default_value,legacy_value)
            assert default_metrics['bitwise_equal'] and u.fingerprint(fp32)==before32
        del fp32,legacy_value,default_value
        model=bridge.load_encoder(args.canonical_spec,device='cuda');before=u.fingerprint(model);rng_before=u.rng_hash(torch,np)
        with torch.inference_mode():
            bulk=model.encode_uint8(raw)
            explicit=torch.cat([model.encode_uint8(raw[i:i+1]) for i in range(32)],dim=0)
            first=model.encode_uint8(raw[:1]);first_gate=u.metrics(torch,first,bulk[:1]);whole=u.metrics(torch,bulk,explicit)
            assert bulk.dtype==torch.float32 and tuple(bulk.shape)==(32,1024)
            assert first_gate['fixed_1e5_allclose'] and whole['bitwise_equal'],'Original firstB1 gate or canonical concat equality failed'
            u.safety_check(pids,identities)
            # Exactly two warmups and five complete external B32 timed calls.
            for _ in range(2):model.encode_uint8(raw)
            walls=[];events=[]
            for _ in range(5):
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                began=time.perf_counter();start.record();value=model.encode_uint8(raw);end.record();torch.cuda.synchronize()
                walls.append((time.perf_counter()-began)*1000);events.append(start.elapsed_time(end));del value
            assert u.fingerprint(model)==before and u.rng_hash(torch,np)==rng_before
        u.safety_check(pids,identities);assert u.cache_stats(args.protected_cache)==baseline
        import statistics
        median=statistics.median(walls)
        u.save(args.output/'report.json',dict(actual_exit_code=0,default_FP32_vs_legacy=default_metrics,
            canonical_API_B32_vs_explicit_N1_concat=whole,original_first_B1_B32_gate=first_gate,
            external_batch=32,actual_encoder_microbatch=1,storage_dtype='float32',compute_autocast='float16',warmups=2,trials=5,
            wall_ms=walls,cuda_event_ms=events,wall_median_ms=median,cuda_event_median_ms=statistics.median(events),
            earlier_FP32_ms=605.7,earlier_bulk_FP16_ms=237.3,speedup_vs_earlier_FP32=605.7/median,
            slowdown_vs_earlier_bulk_FP16=median/237.3,earlier_timings_are_separate_run_references=True,
            model_frozen=True,RNG_unchanged=True,old59_cache_stat_exact=True,old11_stayed_T_at_boundaries=True,
            fullcache_or_training_or_SR_started=False,no_candidate_automatically_adopted=True,at=time.time()))
        code=0
    except BaseException:
        u.save(args.output/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()));raise
    finally:
        if baseline is not None:
            try:after=dict(cache_stat_exact=u.cache_stats(args.protected_cache)==baseline,processes=[u.proc_identity(pid) for pid in pids])
            except BaseException as error:after=dict(error=repr(error))
            u.save(args.output/'after_safety_readback.json',after)
        with (args.output/'outer_exit_code').open('x') as stream:stream.write(str(code)+'\n')

if __name__=='__main__':main()
