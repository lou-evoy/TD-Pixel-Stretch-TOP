# Pixel Stretch TOP — real-time GPU pixel stretch for TouchDesigner (CUDA)

A custom TOP that propagates the last surviving pixel's color across the parts of each
scanline removed by a threshold. The node allows for an optional fade-to-black and adjustable stretch length.

## Demo

<!-- screenshots / GIFs / video go here -->
*Coming soon.*

## Why this one

- **No sort.** The effect is a single segmented inclusive scan over the frame — that's what
  keeps it cheap.
- **Allocate-once, zero-copy**, with built-in fade-to-black and stretch-length controls.

## Getting the node

The compiled plugin isn't distributed in this repo. Precompiled builds will be available to
supporters on **Patreon** *(link coming soon)*. If you'd rather compile it yourself, read on.

## Build it yourself

**Requirements:** TouchDesigner 2025.32050 (validated), CUDA Toolkit 12.8+, Visual Studio
2022/2026 (Desktop development with C++), CMake 3.24+, and an NVIDIA GPU (Turing / RTX 20 or newer).

The TD C++ SDK headers (`TOP_CPlusPlusBase.h`, `CPlusPlus_Common.h`) are not in this repo —
they ship inside TouchDesigner at `<TD install>/Samples/CPlusPlus/CudaTOP`, and `-DTD_SDK_DIR`
must point there (the default below assumes a standard `C:/Program Files/Derivative` install).

Run the commands from the **x64 Native Tools Command Prompt for VS** (Start menu); a normal
PowerShell/cmd won't have `cl` and `nvcc` on `PATH`.

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

This produces `build/PixelStretchTOP.dll`. Copy it to
`%USERPROFILE%\Documents\Derivative\Plugins\` (or run `cmake --build build --target install_to_td`
to copy it there in one step), restart TouchDesigner, then add the node from
**OP Create → Custom → "Pixel Stretch"**.

The default build targets `sm_75`–`sm_120` and needs CUDA 12.8+ for `sm_120` (Blackwell / RTX 50).
For older toolkits or GPUs, override the architectures, e.g.
`-DPS_CUDA_ARCHITECTURES="75-real;86-real;89-real"` (run `nvcc --list-gpu-code` to see what your
toolkit supports).
