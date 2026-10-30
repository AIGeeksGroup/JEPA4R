# Official 3Cube state formal entry

`formal_main.py` is a new opt-in entry for the user's explicit run: accepted
official `cube-triple-play-v0`, state46/goal46/action5 FP32, fresh seed0, global
B1024, GCIQL 1M updates, task alpha1, selected `prefetch1`. Agent math, precision,
optimizer, goals, dataset, and other agent hyperparameters are unchanged.
Only files in this new directory are added. Root owns launch/supervision and
runtime environment acceptance; the entry owner does not launch training.

The fixed official `hyperparameters.sh` state-3Cube line is:

```bash
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
```

The latest user selection extends the execution schedule explicitly:

- Save step1 and 100K..1M: 11 original-format checkpoints, without overwriting.
- Run no evaluation rollout during training. At each original evaluation node
  record the five original task actor-seed draws from the global NumPy RNG.
- After training, restore each checkpoint in node order and execute the exact
  original evaluation block, with each node's saved before-NumPy state. Keep
  the same environment and restore the training tail NumPy state afterward.

Both extensions require explicit flags, whose defaults are false:
`--defer_online_evaluation=true --save_first_eval_checkpoint=true`. The entry
also requires `--save_interval=100000`, `--eval_interval=100000`,
`--log_interval=5000`, `--eval_episodes=50`, and the original CPU evaluation,
one video episode/task, temperature0, GaussianNone, frame_stackNone settings.
No restore is accepted for this fresh run. All remaining GCIQL config fields
must exactly equal the original config with only alpha changed to1.

The original main file SHA256 is
`375fad6375047ff59f3baea408ab7197784c1ec1457588a78a631025e434faee`.
Its bytes remain unchanged. The entry parses that fixed source and modifies
exactly two sites in memory: the original Evaluate-if body becomes the
seed-ledger callback, and the original Save-if guard also includes step1.
Restoring those two sites must make the entire AST equal to the original.
`entry_AST_proof.json` records this check. The loop and original evaluation
body are compiled from source; no parallel copy of either loop is maintained.

The reader is the unchanged old `OrderedInput`, SHA256
`4508ce847694976f8af924215d11985ac4d795093b22d62e8424928346fbdda2`.
The selected profile SHA is
`bdcfec0a47eef5c4913505207c743a655e1afa423f37ad51768f611228e68471`.
The original training dataset instance is copied into the adapter before its
instance sample hook is attached. The seed0 initialization sample1 uses the
original sampler once, without lookahead. Training B1024 uses prefetch1;
step1, log/eval/save nodes and the final batch suppress lookahead. Validation,
evaluation and save hooks explicitly drain before external NumPy use. The
original save callback is used and records real network/Adam counts. Host
NumPy state is saved separately as `numpy_rng_<epoch>.npz`; it is not claimed
to be included in the original `params_<epoch>.pkl`.

Data loading calls the unchanged original env-utils function and forwards
only the accepted train `dataset_path` into the public OGBench API. Its
original val-path derivation, compact loading, environment creation and reset
are reused; the explicit path bypasses downloading. Accepted original NPZ
files are checked against their frozen stat records, without rehashing them.
All original source dependencies and the selected profile are SHA-pinned.

At deferred evaluation, the original restore function loads each saved model
using the final agent as its shape template. The unchanged original block
performs CPU device placement, all tasks, 50 measured episodes plus one video
episode per task, original actions/temperature/success aggregation/video and
CSV logging. Each block's actual after-NumPy state must match its training
seed ledger. No global RNG function is patched in production. Source audit
of this fixed Cube path established that explicit task IDs and GaussianNone
use only the five evaluator seed draws from the legacy global RNG; simulator
randomness uses the environment's separate Generator. This extension is
reported as deferred evaluation, with no rollout or SR claimed for a ledger.

W&B is explicitly passed `mode='offline'`; an environment variable alone is
not relied on. A second offline run is opened for the deferred evaluation
stage because W&B would reject node steps1..1M after training logged1M in the
same run. The actual evaluation metrics remain in original `eval.csv` under
the actual original-main save directory. Root sets TMPDIR inside the run so
the original logger's temporary W&B directory persists.

Root supplies `CUBE_FORMAL_RUN_DIR=<new runtime>` and
`JAX_PLATFORMS=cuda,cpu`. The entry writes only `entry_binding.json`,
`entry_AST_proof.json`, `entry_status.json`, `entry_events.jsonl`, node NumPy
sidecars/seed ledgers, and deferred result receipts there. Actual model save
paths are recorded directly from the mutated original FLAGS.save_dir, without
globbing for the newest directory. Final process exits and resource monitoring
belong to the supervisor. Existence of a ledger/checkpoint is not completion.

The local narrow CPU check is:

```powershell
D:/newconda/python.exe -X utf8 -B coordination/cube_downstream_101732Test_20261017_v1/official_state_formal_v1/test_formal_main.py -v
```

It uses the pinned original main function AST and original Dataset/GCDataset
AST with fake agent/environment/logging plumbing, without JAX or an update
network. It compares the initialization sample, all train/validation batch
bytes, actor seed draws, tail RNG, deferred execution order, checkpoint counts,
and literal step1/5K/100K/1M boundary behavior. On Windows the test explicitly
uses Linux's int64 randint width for the2**32 seed bound; production retains
the original Linux expression. Two checks pass. These validate integration
control flow; existing real CPU/numerical/reload gates remain the evidence
for the unchanged reader. No new GPU/gradient matrix or formal launch was
performed by the entry owner.
