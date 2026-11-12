"""Third-party weights the paper uses, expected under checkpoints/ (or $JEPA4R_CHECKPOINTS); the configs that
point here carry ``weights: ???``. Sources are filled in during integration."""

CHECKPOINTS = {
    "visreg/": "VISReg released weights, ViT-B/16 (configs/method/visreg.yaml)",
    "vjepa21/": "V-JEPA 2.1 released weights, ViT-B/16 and ViT-L/16 for Cube (configs/method/vjepa21.yaml)",
    "page4d/": "PAGE-4D, the decoder backbone (configs/decoder/page4d.yaml)",
    "vggt/": "VGGT, Tab. 4a backbone (configs/decoder/vggt.yaml)",
    "depth_anything3/": "Depth Anything 3, Tab. 4a backbone (configs/decoder/da3.yaml)",
}


def main():
    for path, what in CHECKPOINTS.items():
        print(f"checkpoints/{path:<18} {what}")
    raise SystemExit("TODO(integrate): download sources")


if __name__ == "__main__":
    main()
