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

**Prerequisites**

- TouchDesigner 2025.30000+
- CUDA Toolkit 13.x (12.8+ for Blackwell / RTX 50)
- Visual Studio 2022 or 2026 (MSVC, *Desktop development with C++*)
- CMake ≥ 3.24

**Build (Release)**

From an *x64 Native Tools Command Prompt* (so `cl` and `nvcc` are on `PATH`):

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

Output: `build/PixelStretchTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`,
restart TouchDesigner, and add the node from **OP Create → Custom → "Pixel Stretch"**.
