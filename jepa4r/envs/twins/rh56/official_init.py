"""The BC demonstrations of the RH56 twin: the expert (collect.py, split bc_demo, the opposite camera) from Adroit
Relocate's 100 official initialisations. Each episode's reset moves the object and the target to the official
demonstration's positions (obj_pos, target_pos of rh56_official_task_init.json, written by
jepa4r/data/demos/prepare_official_assets.py); the robot, its drive, the physics, the camera and the point sampling stay
the twin's. Reset seeds: --seed-base + demo index. Writes episodes/ (collect.py's output) and bc_demos_manifest.json.
usage: python -m jepa4r.envs.twins.rh56.official_init --init-source <json> --output <dir> [--count 100]"""

import argparse
import json
from pathlib import Path
import sys

from jepa4r.envs.twins.rh56._runtime import temporary_bindings, write_json


def load_initials(path, seed_base):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {seed_base + r["demo_index"]: r for r in payload["records"]}


def make_bridge(base_class, records, forward):
    class OfficialTaskBridge(base_class):
        def reset(self, seed=None):
            _, info = super().reset(seed=seed)
            row = records[seed]
            env = self.env
            env.model.body_pos[env.obj_bid] = row["obj_pos"]
            env.model.site_pos[env.target_sid] = row["target_pos"]
            forward(env.model, env.data)
            return self.observation(), info

    return OfficialTaskBridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-base", type=int, default=64500000)
    parser.add_argument("--start-demo", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args(argv)
    all_records = load_initials(args.init_source, args.seed_base)
    seeds = list(range(args.seed_base + args.start_demo, args.seed_base + args.start_demo + args.count))
    records = {seed: all_records[seed] for seed in seeds}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import mujoco
    from jepa4r.envs.twins.rh56 import camera_preview, collect

    write_json(
        out / "plan.json", dict(episodes=[dict(split="bc_demo", seed=seed, coverage="expert") for seed in seeds])
    )
    bridge = make_bridge(camera_preview.OppositeCameraBridge, records, mujoco.mj_forward)
    with temporary_bindings((camera_preview, "OppositeCameraBridge", bridge)):
        collect.main(["--plan", str(out / "plan.json"), "--output", str(out / "episodes"), "--camera", "opposite"])
    episodes = json.loads((out / "episodes/manifest.json").read_text(encoding="utf-8"))["auxiliary"]
    selected = [
        dict(
            record,
            npz=str((out / "episodes" / record["npz"]).resolve()),
            official_demo_index=records[record["seed"]]["demo_index"],
            source_id=records[record["seed"]]["source_id"],
        )
        for record in episodes
    ]
    write_json(out / "bc_demos_manifest.json", dict(selected=selected, init_source=str(args.init_source.resolve())))
    print(json.dumps(dict(bc_demos_manifest=str(out / "bc_demos_manifest.json"), selected=len(selected))), flush=True)


if __name__ == "__main__":
    main()
