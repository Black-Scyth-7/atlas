"""One-time GPU setup: make the CUDA-enabled llama-cpp-python find its CUDA DLLs.

Background: we build llama-cpp-python from source against CUDA (see README). The
resulting ggml-cuda.dll depends on the CUDA runtime libraries (cuBLAS, cudart,
nvJitLink). On Windows, CUDA 13 ships those in `<CUDA>\bin\x64`, but
llama-cpp-python's loader uses winmode=RTLD_GLOBAL and only probes `<CUDA>\bin`,
so `os.add_dll_directory` doesn't help. The reliable fix is to place the DLLs in
the SAME directory as ggml-cuda.dll, since a module's own directory is always
searched for its dependencies.

Run this once after `pip install` of the CUDA build:
    python setup_gpu.py

It's CPU-safe to skip: if you run the LLM on CPU (n_gpu_layers ignored), you
don't need this.
"""

import glob
import os
import shutil
import sys

# DLL name patterns ggml-cuda.dll needs (version-agnostic globs).
NEEDED = [
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "nvJitLink_*.dll",
]


def find_cuda_bin() -> str:
    """Locate the CUDA <...>\\bin\\x64 directory holding the runtime DLLs."""
    roots = []
    if os.environ.get("CUDA_PATH"):
        roots.append(os.environ["CUDA_PATH"])
    roots += sorted(
        glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"),
        reverse=True,
    )
    for root in roots:
        for sub in (os.path.join(root, "bin", "x64"), os.path.join(root, "bin")):
            if glob.glob(os.path.join(sub, "cudart64_*.dll")):
                return sub
    raise SystemExit(
        "Could not find a CUDA install with cudart64_*.dll. Install the CUDA "
        "Toolkit (see README) or set CUDA_PATH."
    )


def find_llama_lib_dir() -> str:
    """Locate the installed llama_cpp/lib directory (holds ggml-cuda.dll)."""
    try:
        import llama_cpp
    except ImportError:
        raise SystemExit("llama-cpp-python is not installed in this environment.")
    lib = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    if not os.path.exists(os.path.join(lib, "ggml-cuda.dll")):
        raise SystemExit(
            f"{lib} has no ggml-cuda.dll — this looks like a CPU-only build. "
            "Rebuild llama-cpp-python with CUDA (see README) to use the GPU."
        )
    return lib


def main() -> None:
    src = find_cuda_bin()
    dst = find_llama_lib_dir()
    print(f"CUDA DLLs : {src}")
    print(f"llama lib : {dst}\n")

    copied = 0
    for pattern in NEEDED:
        matches = glob.glob(os.path.join(src, pattern))
        if not matches:
            print(f"  WARNING: no match for {pattern}")
            continue
        for path in matches:
            name = os.path.basename(path)
            target = os.path.join(dst, name)
            # Skip if already present with the same size (idempotent reruns, and
            # avoids file-lock errors when the DLL is mapped by a live process).
            if os.path.exists(target) and os.path.getsize(target) == os.path.getsize(path):
                print(f"  ok (present) {name}")
                copied += 1
                continue
            shutil.copy2(path, dst)
            print(f"  copied {name}")
            copied += 1

    if copied == 0:
        sys.exit("No DLLs copied — check your CUDA install.")
    print(f"\nDone ({copied} files). The GPU build can now load. Test with "
          "`python llm.py`.")


if __name__ == "__main__":
    main()
