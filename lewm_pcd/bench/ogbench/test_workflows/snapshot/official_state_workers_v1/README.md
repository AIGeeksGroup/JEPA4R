# Official state input materialization candidate

Scope: default-off input-only adapter for the unchanged official GCDataset,
state46/goal46 FP32, action5, B1024. No agent, optimizer, sampling probabilities,
reward, model RNG, precision, source file, or training protocol is changed.
No formal training or SR is authorized by this directory.

The four modes are `baseline` (the default, direct original sampler),
`prefetch1` (one pending batch, sequential background reads), `workers2`, and
`workers4` (the same one-batch lookahead with two/four read threads). The last
two modes issue exactly four independent role jobs: current observations,
next observations, value goals, actor goals. Actions and metadata remain
sequential. Worker threads perform only `table[indices]` on frozen NumPy data.

The caller runs the entire original `GCDataset.sample`, including its original
`sample_goals`, index-equality rewards, and the p_aug=0 NumPy random draw. A
shallow copy replaces only Dataset's array indexing with deferred gathers;
original Dataset.sample/get_subset and GCDataset.get_observations are reused.
There is no copy or rewrite of sampler logic and no NumPy RNG use in workers.
This is background data materialization, not a background sampling pipeline.

`sample()` returns batches in original order and plans at most one future
batch. Host NumPy RNG is then ahead by one sample. `committed_rng_state` records
the RNG state after the last consumed training batch. Before validation,
evaluation, checkpoint saving, timing-window end, final end, or any external
NumPy random call, call `barrier()`. It waits for and discards pending work,
restores the complete state from before that unused sample, and clears the
future. A known boundary can instead use `sample(prefetch_next=False)` to avoid
unused work. `close()` performs the same drain/rollback and joins both pools.
External NumPy RNG use while lookahead exists is rejected; it cannot be safely
repaired by silently overwriting the externally consumed stream.

```python
with OrderedInput(train_data, mode='workers2', batch_size=1024) as inputs:
    for step in range(1, bounded_steps + 1):
        boundary = step in validation_save_or_end_steps
        batch = inputs.sample(prefetch_next=not boundary)
        agent, info = agent.update(batch)
        if boundary:
            inputs.barrier()
            # Original validation/evaluation/save code runs only here.
```

The adapter does not change official checkpoint serialization. To claim host
RNG restart equivalence, save/restore the actual NumPy state in a separate
sidecar at a drained boundary and verify the next batch. Merely saving the
agent does not persist NumPy's state.

Unsupported augmentation, frame stacking, mutable/non-flat/non-NumPy tables,
or different state/action dimensions fail explicitly. The one-batch limit
bounds memory. Threads may cost more than these small RAM reads; no speed
improvement is claimed before the real-data A/B measurement.

Allowed new files owned by the input adapter: `ordered_input.py`,
`test_ordered_input.py`, this README, and local CPU evidence/freeze files.
Existing source, agents, samplers, visual adapters and official_state_v1 files
remain read-only. Source reference files were read back by the root agent.

Local CPU command (no JAX/Flax installed locally):

```powershell
python test_ordered_input.py --source-only
```

This executes the exact source AST of Dataset/GCDataset with a flat tree-map
and FrozenDict plumbing substitute. It validates input and boundary logic;
it does not establish real JAX/Flax integration or training equivalence.
The test pins source SHA256
`52c373922a87e1ba8ad8b3b96c3474d5e6385e2ea2c7206930f7a8a315816609`.

Real CPU acceptance, using the target's existing environment:

```bash
JAX_PLATFORMS=cpu "$PY" "$D/official_state_workers_v1/test_ordered_input.py" \
  --impl "$D/source/ogbench/impls/utils/datasets.py"
```

Tests compare complete batch key order, shapes, dtypes, bytes, logical RNG,
real host RNG at validation/save/end boundaries, validation's default RNG
behavior, subsequent external random seeds, next batch after close, actual
two/four role concurrency, caller-only sampling, unsupported cases and read
failure propagation/cleanup. GPU loss/gradient/full-state153/checkpoint checks
and full-data timing remain separate root-owned acceptance gates.
