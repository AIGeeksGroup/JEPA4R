from pathlib import Path
import tomllib


REVISION = "18c09ff8d713494f78a8213792262b910977a65d"
SCATTER_WHEEL = (
    "torch_scatter-2.1.2+pt25cu121-cp311-cp311-linux_x86_64.whl"
)


def test_upstream_contract_pins_sonata_rgb_reader_and_scatter_wheel():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(project["project"]["dependencies"])
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "h5py==3.11.0" in dependencies
    assert "h5py==3.11.0" in requirements
    assert REVISION in dependencies
    assert REVISION in requirements
    assert SCATTER_WHEEL in dependencies
    assert SCATTER_WHEEL in requirements
