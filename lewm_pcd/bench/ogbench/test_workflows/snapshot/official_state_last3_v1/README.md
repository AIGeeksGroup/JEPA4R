# Evaluate only the final three existing checkpoints

`evaluate_last3.py --model-dir <exact old model directory> --ledger-dir <old
formal/runtime> --output-dir <new last3/runtime>` runs only800K,900K,1M. Root
owns stopping the former evaluation and supervising this entry. This entry
does not call official.main, agent.update, or any dataset loader. It requires
the CPU JAX backend; original MuJoCo rendering/video settings remain active.

The frozen formal entry SHA is
`616c2905f802065af10277674f39dde802075ecb9ab4ede40cc785d358a80c92`.
Its `run_deferred_evaluations` method is reused with exactly one in-memory AST
change: `expected` becomes `[800000,900000,1000000]`. Restoring that assignment
must make the complete method AST equal to its frozen source. The original
evaluation block is extracted by the frozen compile helper and executed by
that reused method; neither evaluation loop is copied into a parallel runner.

The public environment API is called with `env_only=True`, followed by the
same bootstrap reset used by official env-utils. No raw NPZ is loaded or
hashed. Exact recorded GCIQL config+alpha1 and zero-shaped46/5 examples create
a CPU restore template. Original restore loads1M as that template, then the
original method restores800K/900K/1M. Each actual restore checks checkpoint
SHA/size against its original sidecar, all153 serialized state leaves/finite,
and real network/Adam counters. No optimizer update is performed.

The original before/after NumPy sidecars and task actor-seed ledgers are read
for these three nodes. Original evaluation remains CPU, five tasks,50 measured
episodes+one video episode/task, temperature0, GaussianNone, max1000 steps.
That is750 statistical episodes plus15 video episodes. The new environment
has an independent Gym Generator: the prior live environment state was not
saved. This preserves the official protocol and logged actor seeds; it does
not reproduce the stopped run's initial scenes. The binding records this.

`FLAGS.save_dir` stays at the old model directory for restore. A thin original
CsvLogger factory routes all evaluation rows to new output `eval.csv` and
refuses an existing new CSV. Old eval.csv is SHA-checked before/after. Offline
W&B is explicit. All new logs, per-node reports, restore checks and process
metadata stay in the new output. Progress states report
`training_updates_in_this_process=0` and `source_training_updates=1000000`;
the inherited method's historical budget assertion is not new training.
The stopped process's real failure/interrupt receipts and old results remain
separate, unchanged. Source and model files are never written by this entry.

Local narrow CPU command:

```powershell
D:/newconda/python.exe -X utf8 -B coordination/cube_downstream_101732Test_20261017_v1/official_state_last3_v1/test_evaluate_last3.py -v
```

Two fake-agent checks establish the single AST selection edit, only three
restore/evaluation nodes, zero update calls, unchanged old CSV, separate new
CSV routing, and accurate progress labels. They do not load a real model or
run a simulator/NN matrix. Root separately runs these same two checks on the
target before actual evaluation.
