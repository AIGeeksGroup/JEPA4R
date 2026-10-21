# RH56 Scripted Policy Archive

Status: `SCRIPTED_UNQUALIFIED`.

This directory archives the RM75 + RH56 scripted policy used by the E5U
diagnostic. The source file SHA-256 is
`CA5D5BA06952EC35226C0EEE67808C632A5927B020AC9C985339D9DB508B337F`.

The policy is a geometry-feedback, finite-state controller. It uses simulated
object geometry and actual hand/object contact feedback; it does not load an
Adroit or DexPolicy checkpoint, replay a demo action table, use an attachment
constraint, or filter seeds. Its action contract is a 13-D candidate action
(normalized XYZ, unit WXYZ, and six active hand targets), which is not the
Adroit or DexPolicy schema.

The archived diagnostic used seed `2111210104` for 200 steps at 20 Hz. It
reported 83 goal frames and task success under the existing cumulative
`goal_frames > 25` criterion. This is a single-seed diagnostic result, not a
formal success rate, a multi-seed stability result, or evidence of real-robot
qualification.

Only the policy source is archived here. The E5U runner change, complete
RM75/RH56 environment, robot/hand assets, configuration, tests, diagnostic
traces, videos, logs, and checkpoints are outside this archive. Consequently,
this file alone is not a standalone runnable environment and must not be used
as an official CortexBench or DexPolicy baseline.
