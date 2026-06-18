/* Pixel Stretch — CUDA algorithm interface.
 * plain C++17 (no kernel syntax) so it builds under MSVC and nvcc.
 * effect: per-line, surviving pixels render as-is; others smear the last survivor's color.
 * segmented inclusive scan, no sort. pipeline: ingest -> scan -> write.
 */
#ifndef PIXELSTRETCH_CUDA_H
#define PIXELSTRETCH_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace pixelstretch {

// max image dimension
static constexpr int kMaxDim = 16384;

enum class Axis : int32_t
{
    Horizontal = 0,   // along rows; line length = width
    Vertical   = 1,   // along columns; line length = height
};

// fill direction: Ascending = survivor before (forward), Descending = survivor after (backward)
enum class Order : int32_t
{
    Ascending  = 0,
    Descending = 1,
};

// thresholding scalar. values MUST match the Threshold Criterion menu order in setupParameters()
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

    // 1 => identity, 0 => blank
    Channel  criterion = Channel::Luminance;
    float    threshold = 1.0f;   // host-clamped [0,1]

    // smear length as fraction of section; beyond it = transparent
    float    stretchLength = 1.0f;  // host-clamped [0,1]

    // 0 = flat hold, 1 = black at section end
    bool     fade       = false;
    float    fadeAmount = 1.0f;  // host-clamped [0,1]

    // BGRA8 (TD default) vs RGBA8; only affects R/G/B mapping for criterion
    bool     bgra      = true;

    // pass-through; API doesn't expose native Bypass to plugin
    bool     bypass    = false;
};

// owns device scan buffers + CUB temp; one per node. lazy realloc only when N grows
class PixelStretcher
{
public:
    PixelStretcher() = default;
    ~PixelStretcher();

    PixelStretcher(const PixelStretcher&) = delete;
    PixelStretcher& operator=(const PixelStretcher&) = delete;

    // inSurf 0 => output cleared. call between begin/endCUDAOperations(). does NOT sync stream
    cudaError_t process(cudaSurfaceObject_t inSurf,
                        cudaSurfaceObject_t outSurf,
                        const Params& p,
                        cudaStream_t stream,
                        const char** outError);

private:
    cudaError_t ensureCapacity(int width, int height, const char** outError);
    void        freeAll();

    int64_t              myCapacity     = 0;
    // forward: last survivor at/before each pixel
    unsigned long long*  myScan         = nullptr;   // [N]
    // backward: next survivor at/after each pixel
    unsigned long long*  myScanBack     = nullptr;   // [N]
    void*                myCubTemp      = nullptr;
    size_t               myCubTempBytes = 0;
};

} // namespace pixelstretch

#endif // PIXELSTRETCH_CUDA_H
