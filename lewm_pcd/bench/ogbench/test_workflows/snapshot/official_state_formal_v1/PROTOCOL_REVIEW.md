# Frozen formal protocol review

No blocking issue found in the reviewed version. This is a source/control-flow
review; Root owns the formal launch and runtime acceptance.

- Entry: `616c2905f802065af10277674f39dde802075ecb9ab4ede40cc785d358a80c92`
- Owner freeze: `a9456e0576291fa84a0a46d3ccbc036a17a865edc0b4dbe851e5dad58e9aad82`
- Launch plan: `763ab9c241d09f0eb23594835d23b6845e0dcb6df5f6179248ae242ca8335edc`
- Supervisor: `c7594f7335036bbc39c7eab5e764537fd89f3f363e6444a59dbe678b95aaa9ea`
- Original main remains byte-identical:
  `375fad6375047ff59f3baea408ab7197784c1ec1457588a78a631025e434faee`.

The latest authorized plan is fresh seed0, B1024, alpha1, exactly1M updates,
accepted official Triple Cube state46/goal46/action5 float32, selected
prefetch1. Save step1 and100K..1M: **11 checkpoints**. No evaluation rollout
runs during training; all11 checkpoints are evaluated afterward. This latest
choice supersedes the earlier interleaved evaluation plan.

The official task command is impls/hyperparameters.sh:630:

```bash
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=1.0
```

The reviewed launch_plan.json contains the complete concrete production argv,
including seed0/train_steps1M/log5000/eval100K/save100K, CPU evaluation,
video1, and both explicit defer/save-first flags. The entry requires the
entire original agent config with only the official task alpha1 override.
Network/Adam/loss/precision remain unchanged: actor3x512 GELU (no actor LN),
V/twin-Q/target-Q3x512 GELU+LN, Adam3e-4, ddpgbc, expectile.9, discount.99,
tau.005. Fresh restore is forbidden. At1M, Adam count must be1000000 and
network.step1000001.

Exactly two original AST sites are changed in memory: evaluation body becomes
the node ledger callback; save guard additionally admits step1. Restoring
these two sites must restore the complete original AST. The original loop
is compiled from pinned source, and the exact original evaluation body is
compiled separately and executed afterward. No parallel training loop or
physics/agent implementation is maintained.

Independent source audit found **five legacy global NumPy seed draws per
original evaluation node**, one np.random.randint(0,2**32) per task, under the
fixed GaussianNone/explicit taskIDs1..5/frame_stackNone protocol. Cube and
arm randomness uses the environment's separate Generator. The only direct
global draw in ManipSpaceEnv.reset is its no-task-ID branch, which evaluation
never takes. CustomMuJoCoEnv, IK, default TimeLimit/OrderEnforcing/PassiveChecker
paths add no global MT draws. Unused SO3/SE3.sample_uniform helpers do not occur
in this fixed call chain.

The entry therefore drains prefetch at original boundaries, records node
before/after NumPy states and those five seeds without rollouts, then saves.
After training it restores each real checkpoint, reuses the same environment,
sets node.before, runs the original evaluation body once, and verifies the
full resulting NumPy state equals node.after. It restores the training-tail
state afterward. Original initialization sample1 remains a single direct
sampler call; validation also retains the original sampler. The copied
deferred view is made before installing the instance sample hook, avoiding
recursive sampling.

Deferred evaluation retains five tasks ×50 statistical episodes plus one
video/task, horizon1000, CPU actor, temperature0, GaussianNone and original
success/CSV aggregation. Total2750 statistical and55 video episodes. These
are original-main evaluations, separate from the prior custom paired Native
protocol. A ledger or checkpoint is not an SR measurement. Separate offline
W&B runs prevent backward checkpoint steps being dropped; TMPDIR is inside
the run. Original environment resets remain unseeded as in source, so a
training seed alone does not establish reproducible per-episode initial states
across separate processes.

Owner local two source-AST/fake-agent boundary tests passed (exit0), and Root's
actual target run passed the same two checks in0.134s (actual exit0, matching
frozen SHA values). Reviewer did not repeat tests/NN/GPU work. Existing real
reader loss/gradient/full153/validation/fresh-restore gates remain reused.

Auxiliary review: cuda,cpu backend visibility, explicit offline logging,
fresh output, runtime-check-before-trainer, freeze verification, persistent
argv/final white-listed env/status/resources/exits, and8GiB owned-PGID RSS
guard are coherent. Only the owned PGID can be signalled; detached services
are outside that declared accounting scope. The approved Temp exception does
not establish a final local backup.

Not established here: actual runtime CUDA gate, stable formal updates,1M
completion,11 model readbacks, deferred EGL/video/physics/SR completion, or
final local backup. PROTOCOL_REVIEW.json records detailed source SHA values,
RNG callsites, exact argv, versions and CPU receipts.
