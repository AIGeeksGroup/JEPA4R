Functional source owner: cube_pipeline_contract_audit. Only this new directory was edited.

Source dependencies on 31435, ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1:

- cube23_distance_gciql_v1/distance_inputs.py SHA b9906dc2f4e7a38826f5ee05031713618a6ce408ee5555302801505452758f90
- cube23_distance_gciql_v1/state_priority_v1/state_runtime.py SHA 623fff33b0c836c6268257689216cf78de8694fbab1107dddbd4bdbfc3ee1956
- cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py SHA e40a2ff52bd0915b7953d9011c65e02c75f9e0b2d0e454b613d18c275f6c13f0
- Original trainer/core/boundary modules are imported in place. Original sample/sample_goals/object_xyz/create_agent AST is checked exactly after the width and membership overlay.

Delivery schema agreed with cache owner:

    purpose: pilot|formal
    group_id: path-safe group name
    cube_count: 1..4
    latent_dim: 768|1024
    encoder_binding: {family,latent_dim,checkpoint_sha256,spec_path,spec_sha256,adapter:{path,sha256}}
    data_binding: original trainer schema, current real split NPZ path/stat/shapes/episodes
    feature_loader_arguments: {train:{cache_dir,split,manifest_sha256,checkpoint_sha256,source_npz,expected_rows},val:{...}}
    camera: {position,actual_quaternion,FOV or fov,...}
    cache_audit: {path,sha256}

Pilot is one real first episode per split, each 1001 raw rows including terminal. The cache owner supplies distinct pilot NPZ and receipts bound to those actual subset stats, with latent hardlinks inside a separate pilot cache root. Formal receipts/files are preserved.

CLIs (prepend the configured Python executable; paths are relative to case root foundation_visual_gciql_v1):

    functional/prepare.py --delivery DELIVERY.json --output CONFIG.json
    functional/runtime.py verify-hooks --config CONFIG.json
    functional/runtime.py gate --config CONFIG.json --output EXEC_ROOT/gates/GROUP
    functional/runtime.py reload --config CONFIG.json --output EXEC_ROOT/gates/GROUP --report EXEC_ROOT/gates/GROUP_reload.json
    functional/native_sr.py --config CONFIG.json --mode tiny --gate-smoke --output EXEC_ROOT/native_sr/GROUP
    functional/runtime.py train --config FORMAL_CONFIG.json
    functional/runtime.py accept --config FORMAL_CONFIG.json --output EXEC_ROOT/runs/GROUP --report EXEC_ROOT/acceptance/GROUP.json
    functional/native_sr.py --config FORMAL_CONFIG.json --mode full --output EXEC_ROOT/native_sr/GROUP
    functional/audit_native.py --config CONFIG.json --mode tiny|full --output NATIVE_OUTPUT

EXEC_ROOT=case/pilot for pilot; case for formal. Pilot configs expose gate/reload/tiny Native commands only, and explicitly refuse train/accept/full Native. Formal train is fresh seed0 B1024 1M, original agent config, async100, same-batch four role threads, validation every5000, eleven checkpoints, no intermediate SR. Native tiny gate smoke reads counter2 and validates the gate and CPU reload fingerprints. Formal Native reads final1M and CPU acceptance, 5 tasks x 50 episodes, caps 200/500/1000/1000.

Formal Native factory examples are read from the run's final_val_probe.npz, already used by CPU acceptance; no repeated formal GPU gate is needed. Independent audit reuses the SHA-pinned original result auditor with current-width substitution only. It checks every episode exit, expected task/index/seed, cap, input widths, 2N encoder counts and success aggregation; additionally it opens every actual steps.npz and checks finite bounded actions/rewards, lengths and final-step success flags. The preserved evaluator adds final-step info to stats (source evaluation.py lines109-115), so no any-step success reinterpretation is introduced. Native captures source/config/spec/encoder checkpoint and downstream checkpoint manifest/params/sidecar/report byte hashes before and after execution; audit freshly verifies those actual bytes against captured bindings. Freeze functional and encoder sources before smoke, since changing them invalidates the source binding.

CPU-only verification completed locally: 3 unittest tests pass, including all eight D/N shapes, unchanged sampling RNG consumption, ALL-distance labels, true terminal-next row, and stage paths. compileall passes. Real remote imports, real cache, GPU update/reload/Native smoke, full training and formal SR have NOT been run by this owner. Root owns their sequencing and launch.
