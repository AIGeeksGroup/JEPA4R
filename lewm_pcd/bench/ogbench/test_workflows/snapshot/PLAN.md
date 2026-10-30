# Cube downstream migration plan

Target: `ssh 101732Test`, `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1`.
Source: `ssh 10155090`; its WM training, data and environments remain unchanged.
User explicitly approved this target's Temp storage.

1. Export immutable official OGBench/GCIQL source, current LeWM backbone and Stable World Model dependency closure, point-cloud/render helpers, configuration and source/version receipts. Transfer only the accepted 2D epoch10 bare checkpoint and latest closed 3D epoch checkpoint, each with its actual config and provenance. Do not relabel the active 3D run as complete.
2. Install a separate environment from the previously validated exact dependency versions, using mirrors. Preserve the target base environment. Limit preparation to two CPU cores, four GiB own process-group RSS, no JAX preallocation and bounded GPU smoke.
3. Add only new explicit Cube-count/path/input adapters: current192+proprio19, goal192+three XYZ coordinates per cube; official sampler/index-equality reward and task alpha1.0. Keep official update/loss/optimizer math unchanged. Retain the already validated optional-example-goal GCIQL factory interface as an unmodified reference.
4. Provide profiles for 1/2/3/4Cube. Triple is the active preparation profile. Other counts are reserved interfaces with explicit unbound weights/data; no collection, training or SR is started.
5. Verify package SHA/member lengths, strict checkpoint loading, frozen encoders, RGB/PCD real forward and small GCIQL GPU updates/save/reload. Use a labelled small fixture if official real data/cache are unavailable. Record that distinction; no formal training or SR and no claim of final 3D readiness.

Success: actual target receipts and exits for migration/runtime/smoke, source and checkpoint SHA/length, persistent logs/configs, and an honest ready/pending inventory. No core refactor, no overwrite of old outputs, no large dataset/old model/whole environment transfer.
