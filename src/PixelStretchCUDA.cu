/* Pixel Stretch — CUDA implementation.
 *
 * Targets CUDA Toolkit 13.x (validated on 13.3.33) and CUB 3.3 (bundled in CCCL).
 * Builds a fat binary across Turing..Blackwell; see CMakeLists.txt for the gencode list.
 *
 * ============================ ALGORITHM =====================================
 * Per line (row for Horizontal, column for Vertical) we forward-fill the last
 * surviving pixel's color across the pixels that don't survive the alpha threshold.
 * This is a SEGMENTED INCLUSIVE SCAN (segments = lines): each scan element carries
 * the packed RGBA color of the most recent survivor at or before it, a "valid" bit
 * (was there a survivor yet in this line), and a "segment head" bit at line starts.
 *
 * We work in "line-order" index space k = 0..N-1, where
 *     line      = k / L       pos = k % L       L = width (H) or height (V)
 *     coord(k)  = (pos, line)  for Horizontal,  (line, pos) for Vertical
 * The same mapping is used for input read and output write.
 *
 * Pipeline:  ingest (surf read -> scan element) -> InclusiveScan -> write (scan -> surf)
 * No sort: three passes plus one scan.
 * ===========================================================================
 */

#include "PixelStretchCUDA.h"
#include "CudaCheck.h"

#include "device_launch_parameters.h"
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/reverse_iterator.h>
#include <algorithm>

namespace pixelstretch {

// --------------------------- scan element layout ----------------------------
// Forward scan element (uint64): bit63 = segment head (line start), bit62 = valid (a
// survivor seen in this line at/before here), bits 32..45 = survivor's within-line
// position (14 bits, <=16383), bits 0..31 = its packed RGBA color.
static constexpr unsigned long long SEG = 1ull << 63;
static constexpr unsigned long long VAL = 1ull << 62;
static constexpr unsigned long long COLOR_MASK = 0xFFFFFFFFull;

static __host__ __device__ __forceinline__
unsigned long long packElem(bool segHead, bool valid, uint32_t pos, uint32_t color)
{
    return (segHead ? SEG : 0ull) | (valid ? VAL : 0ull)
         | ((unsigned long long)(pos & 0x3FFFu) << 32) | (unsigned long long)color;
}

// Classic segmented-inclusive-scan combine. The underlying op is "right survivor
// wins, else carry left". A right element that begins a new segment (line) blocks any
// carry from the left. Associative. The carried "value part" (~SEG) bundles valid +
// survivor position + color, so they all travel together.
struct StretchOp
{
    __host__ __device__ __forceinline__
    unsigned long long operator()(unsigned long long a, unsigned long long b) const
    {
        unsigned long long seg = (a | b) & SEG;
        unsigned long long vp;
        if (b & SEG)            vp = b & ~SEG;   // new segment: ignore left
        else if (b & VAL)       vp = b & ~SEG;   // right is a survivor
        else                    vp = a & ~SEG;   // carry left
        return seg | vp;
    }
};

// The backward scan uses the SAME element format and StretchOp, but with the segment
// head at each line's LAST pixel and scanned over reverse iterators — yielding the NEXT
// survivor (color + position) at/after each pixel. So both fill directions are available.

static __host__ __device__ __forceinline__ int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

static __device__ __forceinline__ void lineOrderToXY(
    int k, int width, int height, Axis axis, int& x, int& y)
{
    if (axis == Axis::Horizontal) { y = k / width;  x = k - y * width;  }
    else                          { x = k / height; y = k - x * height; }
}

static __device__ __forceinline__ float4 toRGBA(uchar4 c, bool bgra)
{
    float r, g, b, a;
    if (bgra) { b = c.x; g = c.y; r = c.z; a = c.w; }
    else      { r = c.x; g = c.y; b = c.z; a = c.w; }
    const float inv = 1.0f / 255.0f;
    return make_float4(r * inv, g * inv, b * inv, a * inv);
}

// Criterion scalar in [0,1]. Matches the Pixel Sort TOP's definitions (Rec.709 luma, HSV).
static __device__ __forceinline__ float channelValue(float4 c, Channel chan)
{
    switch (chan)
    {
        case Channel::Red:        return c.x;
        case Channel::Green:      return c.y;
        case Channel::Blue:       return c.z;
        case Channel::Alpha:      return c.w;
        case Channel::Value:      return fmaxf(c.x, fmaxf(c.y, c.z));
        case Channel::Saturation:
        {
            float mx = fmaxf(c.x, fmaxf(c.y, c.z));
            float mn = fminf(c.x, fminf(c.y, c.z));
            return (mx <= 0.0f) ? 0.0f : (mx - mn) / mx;
        }
        case Channel::Hue:
        {
            float mx = fmaxf(c.x, fmaxf(c.y, c.z));
            float mn = fminf(c.x, fminf(c.y, c.z));
            float d  = mx - mn;
            if (d < 1e-6f) return 0.0f;
            float h;
            if      (mx == c.x) h = fmodf((c.y - c.z) / d, 6.0f);
            else if (mx == c.y) h = (c.z - c.x) / d + 2.0f;
            else                h = (c.x - c.y) / d + 4.0f;
            h *= (1.0f / 6.0f);
            if (h < 0.0f) h += 1.0f;
            return h;
        }
        case Channel::Luminance:
        default:                  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    }
}

// --------------------------------- kernels ----------------------------------

// 1. Read each pixel, decide if it survives the alpha threshold, emit a scan element.
//    survive = alpha(byte) > keptThreshInt, where keptThreshInt is host-mapped so that
//    threshold=1 -> keptThreshInt=-1 (all survive) and threshold=0 -> 255 (none survive).
__global__ void ingestKernel(
    cudaSurfaceObject_t inSurf, int width, int height, Axis axis, bool bgra,
    Channel criterion, int keptThreshInt, bool wantFwd, bool wantBack,
    unsigned long long* __restrict__ scan, unsigned long long* __restrict__ scanBack)
{
    int N = width * height;
    int L = (axis == Axis::Horizontal) ? width : height;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    {
        int x, y;
        lineOrderToXY(k, width, height, axis, x, y);

        uchar4 c;
        surf2Dread(&c, inSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
        uint32_t color = (uint32_t)c.x | ((uint32_t)c.y << 8)
                       | ((uint32_t)c.z << 16) | ((uint32_t)c.w << 24);

        // Survive when the chosen criterion passes the threshold. critByte in 0..255 is
        // compared against keptThreshInt in -1..255 (host-mapped: threshold=1 -> -1 so
        // everything survives; threshold=0 -> 255 so nothing does).
        float crit = channelValue(toRGBA(c, bgra), criterion);
        int critByte = (int)(fminf(fmaxf(crit, 0.0f), 1.0f) * 255.0f + 0.5f);
        bool survive = critByte > keptThreshInt;
        int pos = k - (k / L) * L;

        if (wantFwd)  scan[k]     = packElem(pos == 0,       survive, (uint32_t)pos, color);
        // Backward element: segment head at the line END (first element in reverse).
        if (wantBack) scanBack[k] = packElem(pos == (L - 1), survive, (uint32_t)pos, color);
    }
}

// 3. Write the result: each pixel shows the carried survivor color, or transparent
//    black if no survivor has occurred yet in its line. When 'fade' is set, the held
//    region ramps from the survivor color to black across the section (scanBack gives
//    the section end = next survivor).
__global__ void writeKernel(
    const unsigned long long* __restrict__ scanFwd, const unsigned long long* __restrict__ scanBack,
    Order order, bool haveSecondary, bool fade, float fadeAmount, float stretchLength,
    int width, int height, Axis axis, cudaSurfaceObject_t outSurf)
{
    int N = width * height;
    int L = (axis == Axis::Horizontal) ? width : height;
    const bool asc = (order == Order::Ascending);
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    {
        // Primary scan = the fill source for this direction; secondary = the other
        // boundary (used to size the stretch/fade). Ascending fills from the survivor
        // before (forward); descending from the survivor after (backward).
        unsigned long long prim = asc ? scanFwd[k] : scanBack[k];

        uint32_t out;
        if (!(prim & VAL))
        {
            out = 0u;  // no survivor in the fill direction -> transparent
        }
        else
        {
            uint32_t color  = (uint32_t)(prim & COLOR_MASK);
            int pos         = k - (k / L) * L;
            int survivorPos = (int)((prim >> 32) & 0x3FFFu);
            int d           = asc ? (pos - survivorPos) : (survivorPos - pos);  // >= 0

            if (d == 0 || !haveSecondary)
            {
                out = color;  // the survivor itself, or full-length flat hold
            }
            else
            {
                unsigned long long sec = asc ? scanBack[k] : scanFwd[k];
                int boundary = (sec & VAL) ? (int)((sec >> 32) & 0x3FFFu)
                                           : (asc ? L : -1);          // else the line edge
                int S = asc ? (boundary - survivorPos) : (survivorPos - boundary); // >= 1
                float stretchSpan = stretchLength * (float)S;

                if ((float)d > stretchSpan)
                {
                    out = 0u;  // beyond the stretched length -> transparent
                }
                else
                {
                    float bright = 1.0f;
                    if (fade && stretchSpan > 0.0f)
                    {
                        float t = (float)d / stretchSpan;
                        bright = 1.0f - fadeAmount * t;
                        if (bright < 0.0f) bright = 0.0f;
                    }
                    uint32_t r = (uint32_t)((color & 255)        * bright + 0.5f);
                    uint32_t g = (uint32_t)(((color >> 8) & 255) * bright + 0.5f);
                    uint32_t bl= (uint32_t)(((color >> 16) & 255)* bright + 0.5f);
                    uint32_t a = (color >> 24) & 255;      // keep the survivor's alpha
                    out = r | (g << 8) | (bl << 16) | (a << 24);
                }
            }
        }
        int x, y;
        lineOrderToXY(k, width, height, axis, x, y);
        surf2Dwrite(out, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
    }
}

// Used when no input is connected: clear to transparent black.
__global__ void clearKernel(cudaSurfaceObject_t outSurf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    surf2Dwrite((uint32_t)0u, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
}

// Bypass: copy input straight to output, unaffected (native TOP bypass behavior).
__global__ void passthroughKernel(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 c;
    surf2Dread(&c, inSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
    surf2Dwrite(c, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
}

// ------------------------------ PixelStretcher ------------------------------

PixelStretcher::~PixelStretcher()
{
    freeAll();
}

void PixelStretcher::freeAll()
{
    cudaFree(myScan);     myScan     = nullptr;
    cudaFree(myScanBack); myScanBack = nullptr;
    cudaFree(myCubTemp);  myCubTemp  = nullptr;
    myCubTempBytes = 0;
    myCapacity = 0;
}

cudaError_t PixelStretcher::ensureCapacity(int width, int height, const char** outError)
{
    int64_t N = (int64_t)width * (int64_t)height;
    if (N <= myCapacity)
        return cudaSuccess;  // reuse existing buffers (no per-frame malloc)

    freeAll();

    PS_CUDA_RETURN(cudaMalloc(&myScan,     N * sizeof(unsigned long long)), outError);
    PS_CUDA_RETURN(cudaMalloc(&myScanBack, N * sizeof(unsigned long long)), outError);

    // Temp storage must cover the forward scan and the backward (reverse-iterator)
    // scan; both use uint64 + StretchOp. Allocate the max once.
    size_t fwdBytes = 0, backBytes = 0;
    PS_CUDA_RETURN(cub::DeviceScan::InclusiveScan(
        nullptr, fwdBytes, myScan, myScan, StretchOp(), (int)N), outError);
    auto rb = thrust::make_reverse_iterator(myScanBack + N);
    PS_CUDA_RETURN(cub::DeviceScan::InclusiveScan(
        nullptr, backBytes, rb, rb, StretchOp(), (int)N), outError);

    myCubTempBytes = std::max(fwdBytes, backBytes);
    PS_CUDA_RETURN(cudaMalloc(&myCubTemp, myCubTempBytes), outError);

    myCapacity = N;
    return cudaSuccess;
}

cudaError_t PixelStretcher::process(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
    const Params& p, cudaStream_t stream, const char** outError)
{
    if (outError) *outError = nullptr;

    const int W = p.width, H = p.height;
    const int N = W * H;

    if (!inSurf)
    {
        dim3 b(16, 16, 1);
        dim3 g(divUp(W, b.x), divUp(H, b.y), 1);
        clearKernel<<<g, b, 0, stream>>>(outSurf, W, H);
        PS_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    // Bypass: pass the input through untouched, like a native TOP's Bypass flag.
    if (p.bypass)
    {
        dim3 b(16, 16, 1);
        dim3 g(divUp(W, b.x), divUp(H, b.y), 1);
        passthroughKernel<<<g, b, 0, stream>>>(inSurf, outSurf, W, H);
        PS_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    PS_CUDA_RETURN(ensureCapacity(W, H, outError), outError);

    // Map threshold so survive = alpha_byte > keptThreshInt:
    //   threshold=1 -> keptThreshInt = -1  (alpha > -1 always -> identity)
    //   threshold=0 -> keptThreshInt = 255 (alpha > 255 never -> blank)
    int keptThreshInt = (int)((1.0f - p.threshold) * 256.0f + 0.5f) - 1;

    const int BS = 256;
    int grid = std::min(divUp(N, BS), 65535 * 4);

    // The "primary" scan is the fill direction (forward for Ascending, backward for
    // Descending) and always runs. The "secondary" (opposite) scan sizes the stretch/
    // fade and only runs when needed.
    const bool asc           = (p.order == Order::Ascending);
    const bool needSecondary = p.fade || (p.stretchLength < 1.0f);
    const bool runFwd        = asc ? true : needSecondary;
    const bool runBack       = asc ? needSecondary : true;

    ingestKernel<<<grid, BS, 0, stream>>>(
        inSurf, W, H, p.axis, p.bgra, p.criterion, keptThreshInt, runFwd, runBack,
        myScan, myScanBack);
    PS_CUDA_CHECK_LAUNCH(outError);

    // Forward scan: last survivor (color + position) at/before each pixel.
    if (runFwd)
        PS_CUDA_RETURN(cub::DeviceScan::InclusiveScan(
            myCubTemp, myCubTempBytes, myScan, myScan, StretchOp(), N, stream), outError);

    // Backward scan: next survivor at/after each pixel. Scanning over reverse iterators
    // turns the global left-to-right scan into a per-line right-to-left one (segment
    // heads sit at line ends).
    if (runBack)
    {
        auto rb = thrust::make_reverse_iterator(myScanBack + N);
        PS_CUDA_RETURN(cub::DeviceScan::InclusiveScan(
            myCubTemp, myCubTempBytes, rb, rb, StretchOp(), N, stream), outError);
    }

    writeKernel<<<grid, BS, 0, stream>>>(
        myScan, myScanBack, p.order, needSecondary, p.fade, p.fadeAmount, p.stretchLength,
        W, H, p.axis, outSurf);
    PS_CUDA_CHECK_LAUNCH(outError);

    return cudaSuccess;
}

} // namespace pixelstretch
