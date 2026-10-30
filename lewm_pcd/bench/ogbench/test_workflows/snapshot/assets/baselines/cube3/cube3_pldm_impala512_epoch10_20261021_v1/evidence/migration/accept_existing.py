"""Reuse frozen acceptance; select the recorded own run instead of its parent."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_own_run(tree, own):
    """Insert exactly one filter; preserve every original rejection/check."""
    count = 0

    class Patch(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            nonlocal count
            if node.name != "accept":
                return node
            body = []
            for statement in node.body:
                body.append(statement)
                if (isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "runs"
                        and isinstance(statement.value, ast.SetComp)):
                    body.extend(ast.parse(
                        f"runs = {{x for x in runs if Path(x).resolve() == Path({str(own)!r}).resolve()}}"
                    ).body)
                    count += 1
            node.body = body
            return node

    result = Patch().visit(tree)
    if count != 1:
        raise ValueError(f"expected one original run-filter boundary, got {count}")
    return ast.fix_missing_locations(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    formal = root / "outputs/pldm_native_prefetch_selected_continuation_v1"
    training = formal / "formal/pldm"
    read = lambda path: json.loads(Path(path).read_text())
    observed = read(training / "result/resume_readback.json")
    own = Path(observed["own_run"]).resolve()
    if not own.is_relative_to(formal / "_spt_cache/runs"):
        raise ValueError("own run is outside this formal run")
    if Path(read(own / "run_meta.json")["run_dir"]).resolve() != own:
        raise ValueError("own-run metadata differs")
    if (training / "actual_exit_code").read_text().strip() != "0":
        raise ValueError("completed trainer required")
    if args.report.exists():
        raise FileExistsError(args.report)
    sys.path.insert(0, str(root))
    native_path = root / "native_prefetch_continuation_v1/continuation.py"
    if sha(native_path) != "d686d58044e8ff8875a2e0f1289488e1d6e7aba03244dbddd44f533e40e04e2c":
        raise ValueError("native continuation source differs")
    spec = importlib.util.spec_from_file_location("acceptance_original_native", native_path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    a = SimpleNamespace(root=root, mode="accept", training_output=training,
        output=formal / "acceptance_migration_v1/pldm/result",
        release=root / "root_authorizations/native_prefetch_continuation_full_release_v1.json",
        parent_proof=root / "outputs/native_prefetch_boundary_epoch4_v1/preserved_epoch4/status.json",
        trial_report=root / "outputs/native_prefetch_trial_v1/status.json")
    _, _, parent, _, _ = native.require_runtime(a)
    api = native.api()
    original_tree = api.acceptance_tree
    api.acceptance_tree = lambda source: select_own_run(original_tree(source), own)
    import torch
    torch.set_num_threads(1)
    value = api.run_accept(a, parent)
    value.update(actual_exit_code=0, accepted=True,
        acceptance_recovery="explicit resume_readback own-run filter only",
        original_acceptance_exit_code=1,
        original_failure_preserved=True, source_helper_sha256=sha(__file__),
        native_source_sha256=sha(native_path),
        acceptance_source_freeze_sha256=sha(native_path.parent / "source_freeze_v1.json"),
        GPU_used=False, model_forward_performed=False,
        raw_dataset_opened=False, RNG_exact_continuation=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    print(json.dumps({k: value[k] for k in (
        "accepted", "actual_exit_code", "global_step", "Adam_steps",
        "strict_factory_reload", "full_bare_exact", "new_complete_validation_epochs")}))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
