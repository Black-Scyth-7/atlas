"""Create an isolated GPU/CUDA venv (.venv-gen) for GENERATING wake-word audio.

piper-sample-generator needs `piper` + torch. The main .venv has piper but only
CPU torch; .venv-train has CUDA torch but no piper. This builds a dedicated env
with a Blackwell-capable CUDA torch AND piper so 100k clips generate on the GPU
in minutes. Kept separate so it can't disturb the assistant or training envs.

    python setup_gen.py

Then generate more "Atlas" positives (see README / the printed hint below):
    .venv-gen\\Scripts\\python -m piper_sample_generator "Atlas" ^
        --model wake_training\\piper-sample-generator\\models\\en_US-libritts_r-medium.pt ^
        --max-samples 100000 --batch-size 100 ^
        --output-dir wake_training\\atlas_data\\positive_extra
(run from the piper-sample-generator folder, or set PYTHONPATH to it).
"""

import os
import subprocess
import venv

VENV = ".venv-gen"
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

    print("\nInstalling generator deps (piper-tts, audiomentations, ...)...")
    if pip("install", "-r", "requirements-gen.txt"):
        raise SystemExit("generator deps install failed.")

    # piper-tts pulls a generic torch; re-pin the CUDA build (no deps) so the
    # GPU one wins.
    pip("install", "--force-reinstall", "--no-deps",
        "torch", "torchaudio", "--index-url", CUDA_INDEX)

    print("\nVerifying GPU...")
    subprocess.run([py, "-c",
        "import torch; print('torch', torch.__version__, '| cuda', "
        "torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() "
        "else 'NO GPU VISIBLE'); import piper; print('piper OK')"])

    gen = os.path.join("wake_training", "piper-sample-generator")
    print("\nDone. Generate 100k 'Atlas' clips on the GPU with:")
    print(f'  set PYTHONPATH={gen}')
    print(f'  {py} -m piper_sample_generator "Atlas" '
          f'--model {gen}\\models\\en_US-libritts_r-medium.pt '
          '--max-samples 100000 --batch-size 100 '
          '--output-dir wake_training\\atlas_data\\positive_extra')


if __name__ == "__main__":
    main()
