# G01 / G04 / G07 final 1M SR

- Host: `101732Test`; original training controller and all checkpoints stay unchanged.
- Bind each accepted `update_001000000`, original GCIQL restore and deterministic `sample_actions`; true current/goal widths 28/3, 37/6, 46/9; no padding.
- Reuse pinned official `evaluate`, task reset, action clipping/control, termination and success. The only policy interface projects full goal to cube XYZ.
- Five tasks × 50 metric episodes per group; original limits 200/500/1000. Fixed reset seed `970000000 + task_id*100000 + episode_index`; no added warmup or video episodes. Preserve original reset's action-space RNG behavior and record the limitation; do not claim identical historical visual scenes.
- Run one short real-environment diagnostic per model, then formal groups and independent metadata audits serially. Diagnostics are excluded from SR. No updates, data/cache production or performance matrix.
- Reuse the original process-group resource guard; save actual argv, phase logs, resources, actual exits and group cleanup. Stop the owned job on actual failure, without automatic retry.
- Publish comparison only after 250 unique metric episodes per group, all task/phase/audit/outer exits 0, checkpoint/source bindings and closed evidence local copy verified.
