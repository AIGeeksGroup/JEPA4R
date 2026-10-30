# Official Triple state GCIQL: bounded smoke

Only this new directory is changed. All original agent, sampler, reward, save/restore and visual adapters remain unchanged. The state baseline uses current46/goal46 FP32, action5, without zero padding, a visual mask or an encoder. Official Triple overrides alpha1.0 and eval50; other official state-profile defaults remain B1024/1M, Adam3e-4, 512x3/LN, discount.99/tau.005/expectile.9.

The target currently has eight actual paired source observations, not the complete official train/val dataset. The fixture's final terminal is artificial. Baseline smoke runs B1024 two updates, same-fixture validation, official save and a distinct `--fresh` invocation restoring all153 state leaves with Adam2/network3 and finite actions. A separate `--bench` run performs64 warmup+3x256 timed updates, synchronizing all agent/info leaves at each window boundary. The external supervisor records GPU utilization and resources against phase.json; fixture diversity and cache locality limit performance conclusions.

Examples after a resource/source check (these commands are not executed by this plan):

```sh
python -B official_state_v1/smoke.py --out /absolute/new_baseline_output
python -B official_state_v1/smoke.py --out /absolute/new_baseline_output --fresh
python -B official_state_v1/smoke.py --out /absolute/new_benchmark_output --bench
```

State is continuous FP32; casting it to uint8 would change the experiment. Unlike images, this batch has only about0.75MB across the four state tensors. No uint8 branch is introduced. A future resident-FP32 table/gather optimization would require actual profiling and sampler/RNG/state equivalence; it is outside this patch.

`run_official_state.sh` previews a later official run and requires explicit existing train/val NPZ files plus a new output path. Only its `--execute` branch forwards the dataset path into the original loader and invokes the unchanged main.py with the official Triple task flags. Official main evaluates all tasks after update1 and periodically; it is not used for the bounded smoke. No formal training or SR is started by preparing these files.
