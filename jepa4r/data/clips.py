"""World-model clips (Sec. 3): ``num_frames`` (= history + 1 = 4) frames spaced ``frameskip`` (= 5) env steps
apart; the action block between consecutive frames concatenates the 5 actions. Observation = point cloud
(JEPA4R) or image (LeWM, PLDM, LeJEPA) from the same rows.
"""


class ClipDataset:
    def __init__(self, data_dir, method_cfg, split="train"):
        raise NotImplementedError("TODO(integrate): clip sampler on top of jepa4r.data.storage")
