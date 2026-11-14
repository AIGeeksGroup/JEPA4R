"""The process environment of the Cube policy stages, set before JAX / Torch are imported: GCIQL training runs JAX on
the GPU; encoding and the online evaluation run JAX on the CPU (MuJoCo rendering through EGL)."""


def runtime_environment(stage, cuda_visible_devices="0"):
    if stage not in ("encode", "gciql", "evaluate"):
        raise ValueError("unknown Cube policy stage: " + str(stage))
    environment = dict(
        CUDA_VISIBLE_DEVICES=cuda_visible_devices,
        JAX_PLATFORMS="cuda,cpu" if stage == "gciql" else "cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_THREEFRY_PARTITIONABLE="false",
        JAX_DEFAULT_PRNG_IMPL="threefry2x32",
        JAX_ENABLE_X64="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        XLA_FLAGS="--xla_gpu_autotune_level=0",
    )
    if stage != "gciql":
        environment["MUJOCO_GL"] = "egl"
    return environment
