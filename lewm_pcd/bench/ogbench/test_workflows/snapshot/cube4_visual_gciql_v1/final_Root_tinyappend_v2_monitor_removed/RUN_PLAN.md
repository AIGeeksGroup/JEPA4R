# 4Cube cached-visual GCIQL Implementation Plan

> For agentic workers: execute the isolated tasks with existing subagents. Each file has one owner; publish and launch only after the relevant freeze and narrow acceptance.

**Goal:** Run G11 RGB and G12 PCD on the official Quad data using the existing accepted GCIQL algorithm and cached192+proprio inputs.

**Architecture:** Reuse the original sampler, readonly RAM feature loader, GCIQL factory/update/save/restore, async100 metrics and owned-process serial supervisor. New files only provide the N4 input guard/factory registration, a verified post-migration stat overlay, per-role configurations and dispatch. Start the ready RGB route first; PCD must produce and independently accept its own Quad cache before G12.

**Tech stack:** Existing 103032Test env_runtime; JAX/Flax/NumPy GCIQL, Torch only for the separate frozen offline PCD producer.

## Global constraints

- Host: ssh103032Test. Remote training namespace: `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1`.
- Official data: cube-quadruple-play-v0, train5000/val500 episodes, 1001 raw states each, state55/action5/qpos42/qvel38. Preserve true terminal1000 and rawID layout.
- Both roles: freshseed0, B1024, 1M updates, original alpha1/Adam3e-4/512x3/LN/.99/.9/tau.005, original FP32 arrays and JAX default precision. No zero filling; no runtime encoder during RL training.
- Current/next `[cache192, proprio19]` =211; each value/actor goal uses its own original sampleID `[cache192, objectXYZ12]` =204. Raw current object XYZ is only for reward/mask.
- Success for training: each of four object L2 distances<=.4 scaled units (4cm), then ALL; reward=success-1, mask=1-success. Original sampler draws/order and validation interleaving remain unchanged.
- Original four same-batch role lookup threads and async100. No worker/precision/sampling matrix; no DataLoader, raw image path or uint8 embedding quantization.
- Checkpoints: step1 plus each100K, 11 full transactions. Every5K one original validation batch, 200 CSV/val records.
- No edits to old sources/results/cache/receipts/checkpoints. No push, no collection, no WM retraining, no SR in this task. GPU stages strictly serial; stop only owned process groups on actual failure/OOM.

## Tasks and ownership

1. **Asset and migration binding — cube2_official_data_inventory**
   - Reuse original source/checkpoint/cache SHA acceptance. Prove necessary current official NPZ bytes match original hashes after inode/ctime changed.
   - New `migration_rebind_v1` sidecar binds old/current stats, source SHA and before/after stability. Do not rewrite old accepted_stat or disable strict guards. Keep original per-NPY SHA/finite checks when the normal loader loads RAM.
   - RGB is not marked ready until original producer/audit/outer exits0 and new migration proof are accepted.

2. **N4 thin interface — cube2_embedding_pipeline_audit**
   - New `functional_v1/quad_visual.py` and focused tests, plus G11/G12 config templates.
   - SHA-pin b990 DistanceInputs and a973 extension runtime; extend only N4 guard/role factory/registration and verified in-memory migration stat projection.
   - Verify normal N2/N3 path unchanged, N4 211/204, per-object ALL reward, same IDs and terminal999→1000, and host RNG byte equality. Freeze before remote release.

3. **PCD cache preparation — double_ckpt_provenance**
   - Reuse accepted PCD producer/renderer/preprocessor/atomic receipts/resume/auditor.
   - New Quad profile only. Bind actual Quad3D epoch10, training up5+localZback10cm camera and actual training crop; do not guess from Double/Triple defaults.
   - CPU source tests first; necessary actual frozen-encoder/preprocess/tiny producer proof only after G11 GPU training and acceptance have exited. No new worker or precision tuning.

4. **Dispatch and monitoring — Root**
   - Reuse original owned-process resource guard and serial gate→CPUreload→freshformal→CPUaccept stages. Formal must never start from diag2 weights.
   - G11: true B1024 two-update/val/save gate0; independent CPU full153/RNG/nextbatch/action reload0; then fresh formal1M; original full acceptance0.
   - G12 only after its actual Quad cache acceptance and corresponding narrow gate. It receives a separate fresh run/output and the same1M budget.
   - Monitor actual counts, CPU/I/O and cgroup memory; conservative RSS24GiB per owned group, actual OOM events stop that group. No silent fallback or hyperparameter change.
   - Estimate each long stage from actual submitted windows and set one check about10 minutes before the estimated end; no fixed10/15-minute polling.

5. **Closure**
   - Per-role trainer/gate/reload/accept and parent exits0, 1M/network1000001/Adam1M, 11 transactions and 200 validation rows.
   - Preserve source/config/argv/status/log/exits and necessary final model evidence. Archive/copy closed artifacts once with SHA/length/memberCRC; no repeated source data, embeddings or old WM transfer.
   - This plan does not authorize or claim new Native SR results.

## Current execution (2026-10-20)

- Both Quad LeWM epoch10 weights current-host whole-SHA verified.
- RGB original full producer/audit0 and current migration byteproof accepted; normal G11 loader verified every current feature SHA/finite/ID. New host admission is recorded separately from old producer pending-Root labels.
- G11 actual B1024 gate2+val+save and independent CPU153/nextbatch/RNG reload0, owned groups drained. Formal freshseed0 is running; observed297500/network297501/59val at20:58:51 BJT.
- Original PCD adapter target CPU materialization0. One serial eight-stage controller is armed in pure-stdlib wait for G11 acceptance. No PCD GPU stage has run yet.
- G12 original false template remains false; new metadata helper is frozen/CPU2 and narrow independent static review0. Its actual executable config can only be generated from complete PCD audit and G11 full acceptance.
- One monitor created, first21:44 BJT based actual G11~208Hz windows/conditional21:55 training finish. Acceptance and later stages extra.
- No new Native SR or final acceptance/backup completion is claimed.

G12 after-cache persistent dispatcher9e45 source/49012freeze/caf818handoff: author2fakeCPU0, target2fakeCPU+help0, narrowreviewbcbf0. Unique21:19:16arm;21:19:49actual CPUwaiting/noNN. CurrentG11=562100/network562101/112val/CSV, recordedfinitegrad>0, five5Kpool~211Hz; noOOM/GPUonlyoriginalG11. It generates no executable G12 until realPCD closure and then reuses original four phases.

## 21:55 check: G11 complete and engineering-only PCD recovery

G11 21:54:27 train1M and originalCPUfull153/11txn/200CSV accepted0; all4role exits+wrappers0, host1M sampler replayFalse explicitly. Closed final-only archive42372006B/SHA9176e27a.../73members and one actuallocalcopy+fullCRC/SHA/size/sourcebeforeafter0, Root independent wholeSHA readback0. Do not repeat.

Old PCD/runtime_v1 and old G12waiter actual1 preserved: numeric exact, GPUpreproc only3.9868% faster below old5% engineering gate. Newpolicycc5b/decisionf2d... only two performance guards; numeric/source/renderer/model unchanged. Newcontrol_v2 remaining4planf6f1, controllerfc4a, actual22:10 dispatch; profile/pilot0 then full. NewG12 recovery e620/3977+finalbindingbb47 are deployed CPU2+realbindingload0/independentreview0 and armed pureCPUwait at22:12. Only after real5500EPcacheaudit+new4closed0 can receipt+v3config produce and original G12 four stages begin.

Latest22:25:50 full907EP/907907rows, GPU99%5221MiB/noOOM/18owned16renderer. Twenty/forty/eighty currentcommit windows983.09/979.72/1002.24rows/s conditional encode23:42–44 plus unknownseal/audit. Single next23:32, no fixedpoll. G12 final selectorv3 e420 deployed CPU2/review0 but notexecuted. PCD smallmetadata pack still onlyscope, must implement afterclosure without rereading oldraw or repeating newvectoraudit.

## 23:41 check completed: cache accepted, G12 genuinely training

PCD full5500/5505500 and independent audit0, new4/ownedempty/outerwrapper0; manifest51ae/seal602f/audita1b07/recovery499268. Originalproducer/audit status actualNone historical fields preserved; actual report/exits0 are authoritative. G12 auto publisher0/prepare0 ->actualcfg58b1/planba11 ->B1024 gate2+CPU153reload0 ->formal23:42:37 fresh0/network1/Adam0. Actual23:55:15 G12=139000/network139001/27CSVval, allrecordedfinitegrad>0/noOOM, 92pins exact; recent3/5/7/12 windows208/206/205/196Hz conditionalfinish01:04–08 extraCPUaccept. Single next00:52 Oct21.

PCD metadata v2 selected5642files and packed5643members,15361328B/SHAb50d765d..., sourcebeforeafter/memberSHA/CRC/builder+rootouter0/singleSCPscratch/projectSHA-size0, RootindependentwholeSHA/5500receipt/no binary/activeG12logs0. Originalv1 staticreview1 preserved; only oldexecuted4/new4 pathselection fixed. All source/model/vertexcache original data unchanged. Now remaining onlyG12 finalCPUaccept+unique finalmodelbackup; no newSR.

## Final completion — 2026-10-21

G12 native1M at01:03:40, CPUfull153/Adam1M/network1000001/200CSV/11txn/actionfinite0 at01:03:50, fourrole and threehelper stages/outer/wrappers all0/ownedempty. Uniquee420 final-only package42346081B/107members/SHA4ba92391... and one actualcopybcb verified allmemberSHA/CRC/sourcebeforeafter/wholeSHA-size0, Root again verified fullarchive and two finalpayloads/11manifestSHA0. Missing local-only actualCPUdeploy receipt was rawbyte mirrored8665... after preflight1; e420/model/core unchanged, originalfailure preserved. G11/PCD old packages reused without recopy. Allauthorizedtraining/acceptance/necessarybackups achieved; noSR/newbudget. FinalRootreceipt stored, monitor removal follows actual final tinyappend.

Final app monitor removal returned deleteStatus=deleted; active automation.toml is absent. Final receipt_v2 links all3copy receipts and independent summary. No further automatic training or SR.
