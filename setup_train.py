"""Create an isolated GPU/CUDA venv (.venv-train) for wake-word training.

Keeps CUDA-enabled torch OUT of the main .venv (which the assistant uses with a
CPU torch for SpeechBrain). Installs a Blackwell-capable CUDA torch plus
openWakeWord's training deps, then verifies the GPU is visible.

    python setup_train.py

Then train on the GPU with:
    .venv-train\\Scripts\\python wake_training\\train_atlas.py
"""

import os
import subprocess
import venv

VENV = ".venv-train"
# CUDA 12.8 wheels — support Blackwell (RTX 50-series, sm_120).
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def _venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def main() -> None:
    py = _venv_python(VENV)
    if not os.path.exists(py):
        print(f"Creating {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    else:
        print(f"{VENV} already exists — updating.")

    def pip(*args) -> int:
        return subprocess.run([py, "-m", "pip", *args]).returncode

    pip("install", "--upgrade", "pip")

    print("\nInstalling CUDA torch + torchaudio (Blackwell sm_120)...")
    if pip("install", "torch", "torchaudio", "--index-url", CUDA_INDEX):
        raise SystemExit("CUDA torch install failed — see pip output above.")

    print("\nInstalling openWakeWord training deps...")
    if pip("install", "-r", "requirements-train.txt"):
        raise SystemExit("training deps install failed.")

    # A dep may have pulled the CPU torch; re-pin the CUDA build (no deps).
    pip("install", "--force-reinstall", "--no-deps",
        "torch", "torchaudio", "--index-url", CUDA_INDEX)

    print("\nVerifying GPU...")
    subprocess.run([py, "-c",
        "import torch; print('torch', torch.__version__, '| cuda', "
        "torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() "
        "else 'NO GPU VISIBLE')"])

    print("\nDone. Train on the GPU (50k steps by default) with:")
    print(f"  {py} wake_training\\train_atlas.py")


if __name__ == "__main__":
    main()
