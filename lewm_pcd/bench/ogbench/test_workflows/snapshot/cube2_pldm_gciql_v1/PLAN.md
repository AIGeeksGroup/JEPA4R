# Execution contract

1. Verify source/target checkpoint identity and Python 3.12 compatibility.
2. Prepare official Cube2 metadata from the existing accepted source arrays.
3. Run a two-episode embedding pilot and verify terminal boundaries and B32/B1 alignment.
4. Encode all 1100 episodes, audit every latent and raw-ID contract.
5. Publish the executable B1024 config, run the two-update gate and strict gate reload.
6. Train fresh seed0 for exactly 1,000,000 updates with 11 checkpoints.
7. Accept the final policy on CPU, including all 153 leaves and action forward.
8. Run Native tiny, audit it, then run and audit 250 formal episodes.

Every phase records argv, environment class, stdout/stderr, process state, status and actual exit code. A failed phase stops the serial pipeline and preserves all evidence.
