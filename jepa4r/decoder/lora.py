"""Stage B (Sec. 4.4): rank-4 LoRA [hu2022lora] on the query and value projections of all attention blocks of
the backbone (72 on PAGE-4D, 0.88M parameters).
"""


def add_lora(backbone, rank=4, targets=("q", "v")):
    raise NotImplementedError("TODO(integrate)")
