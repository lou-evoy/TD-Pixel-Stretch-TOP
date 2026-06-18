/* Pixel Stretch — CUDA algorithm interface.
 *
 * Plain C++17 contract between the TouchDesigner glue (PixelStretchTOP.cpp) and the
 * CUDA implementation (PixelStretchCUDA.cu). No CUDA kernel syntax, so it compiles
 * under MSVC as well as nvcc.
 *
 * Effect: along each scanline (row for Horizontal, column for Vertical), a pixel
 * "survives" when its ALPHA passes the threshold; surviving pixels render as-is, and
 * every non-surviving pixel holds the full color of the last survivor before it
 * (a "stretch"/smear). Before the first survivor in a line the output is transparent.
 * Threshold T: 1 => everything survives (identity), 0 => nothing survives (blank).
 *
 * Implemented as a single segmented inclusive scan that carries the last survivor's
 * packed color — no sort. Pipeline: ingest -> scan -> write.
 */
#ifndef PIXELSTRETCH_CUDA_H
#define PIXELSTRETCH_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace pixelstretch {

// Generous upper bound on either image dimension (TD textures cap well below this).
static constexpr int kMaxDim = 16384;

enum class Axis : int32_t
{
    Horizontal = 0,   // stretch along each row;    line length = width
    Vertical   = 1,   // stretch along each column; line length = height
};

// Fill direction along the axis. Ascending: a held pixel holds the survivor BEFORE it
// (forward smear). Descending: it holds the survivor AFTER it (backward smear).
enum class Order : int32_t
{
    Ascending  = 0,
    Descending = 1,
};

// Scalar a pixel is thresholded on to decide if it survives. Integer values MUST match
// the Threshold Criterion menu order in PixelStretchTOP::setupParameters().
enum class Channel : int32_t
{
    Luminance = 0, Hue = 1, Saturation = 2, Value = 3,
    Red = 4, Green = 5, Blue = 6, Alpha = 7,
};

struct Params
{
    int32_t  width     = 0;
    int32_t  height    = 0;
    Axis     axis      = Axis::Horizontal;
    Order    order     = Order::Ascending;

    // A pixel survives when its 'criterion' value passes the threshold; non-surviving
    // pixels are dropped (treated as transparent) and stretched over by the last
    // survivor. 1 => everything survives (identity), 0 => nothing survives (blank).
    Channel  criterion = Channel::Luminance;
    float    threshold = 1.0f;   // clamped to [0,1] on host

    // How far the survivor's color stretches into each held section, as a fraction of
    // the section length. 1 = fill the whole section, 0 = no stretch (only survivors);
    // beyond the stretched length the section is transparent.
    float    stretchLength = 1.0f;  // clamped to [0,1] on host

    // Fade each held section from the survivor's color to black across its length.
    // fadeAmount: 0 = flat hold (no fade), 1 = reaches black at the end of the section.
    bool     fade       = false;
    float    fadeAmount = 1.0f;  // clamped to [0,1] on host

    // True when the texture is BGRA8 (TD's preferred 8-bit format), false for RGBA8.
    // Only affects which channel maps to R/G/B when computing the criterion.
    bool     bgra      = true;

    // When true, copy the input straight to the output (no stretch) — mirrors a native
    // TOP's Bypass flag, which the C++ TOP API does not expose to the plugin.
    bool     bypass    = false;
};

// Owns the device scan buffer + CUB temp storage. One instance per TOP node.
// Allocation happens lazily and is only redone when the pixel count grows (resolution
// change) — never per frame.
class PixelStretcher
{
public:
    PixelStretcher() = default;
    ~PixelStretcher();

    PixelStretcher(const PixelStretcher&) = delete;
    PixelStretcher& operator=(const PixelStretcher&) = delete;

    // Runs the effect on 'stream'. 'inSurf' may be 0 (no input) -> output cleared.
    // Must be called between TOP_Context::beginCUDAOperations()/endCUDAOperations().
    // Does NOT synchronize the stream (TD manages CUDA<->Vulkan ordering).
    cudaError_t process(cudaSurfaceObject_t inSurf,
                        cudaSurfaceObject_t outSurf,
                        const Params& p,
                        cudaStream_t stream,
                        const char** outError);

private:
    cudaError_t ensureCapacity(int width, int height, const char** outError);
    void        freeAll();

    int64_t              myCapacity     = 0;
    // Both scans use the same packed layout (segHead|valid|survivorPos|color).
    // Forward: last survivor at/before each pixel (segHead at line start).
    unsigned long long*  myScan         = nullptr;   // [N]
    // Backward: next survivor at/after each pixel (segHead at line end, reverse scan).
    unsigned long long*  myScanBack     = nullptr;   // [N]
    void*                myCubTemp      = nullptr;
    size_t               myCubTempBytes = 0;
};

} // namespace pixelstretch

#endif // PIXELSTRETCH_CUDA_H
