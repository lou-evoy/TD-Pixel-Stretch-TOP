/* Pixel Stretch TOP — TouchDesigner SDK glue.
 *
 * Mirrors the shipped CudaTOP sample structure (Samples/CPlusPlus/CudaTOP):
 *   - executeMode = TOP_ExecuteMode::CUDA
 *   - acquire input texture as a cudaArray via OP_TOPInput::getCUDAArray()
 *   - create the output cudaArray via TOP_Output::createCUDAArray()
 *   - wrap both as cudaSurfaceObjects, do work between begin/endCUDAOperations()
 *   - one instance-owned cudaStream_t for the node's lifetime
 */
#include "PixelStretchTOP.h"

#include <cassert>
#include <cstdio>
#include <algorithm>

// Single source of truth for the plugin version. Bump on each release; keep the numeric
// parts in sync with customOPInfo.major/minorVersion below.
static const char* kVersion = "1.0.0";

extern "C"
{

DLLEXPORT void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;

    info->executeMode = TOP_ExecuteMode::CUDA;

    info->customOPInfo.opType->setString("Pixelstretch");
    info->customOPInfo.opLabel->setString("Pixel Stretch");
    info->customOPInfo.opIcon->setString("PXT");
    info->customOPInfo.authorName->setString("SAT");
    info->customOPInfo.authorEmail->setString("levoy@sat.qc.ca");

    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;

    info->customOPInfo.majorVersion = 1;
    info->customOPInfo.minorVersion = 0;
}

DLLEXPORT TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new PixelStretchTOP(info, context);
}

DLLEXPORT void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context* context)
{
    delete (PixelStretchTOP*)instance;
}

} // extern "C"

// Bind a fresh surface object to 'array'. We destroy any previous object and create a
// new one every cook rather than caching it: when the node is bypassed and reactivated
// TD frees the old cudaArray and re-registers its interop, so a cached surface handle
// goes stale and probing it raises a sticky cudaErrorInvalidResourceHandle. Recreating
// each cook is only a few microseconds and is robust.
static void
setupCudaSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
    if (*surface)
    {
        cudaDestroySurfaceObject(*surface);
        *surface = 0;
    }
    cudaResourceDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;
    cudaCreateSurfaceObject(surface, &desc);
}

static bool
isSupported8BitRGBA(OP_PixelFormat f)
{
    return f == OP_PixelFormat::BGRA8Fixed || f == OP_PixelFormat::RGBA8Fixed;
}

PixelStretchTOP::PixelStretchTOP(const OP_NodeInfo* info, TOP_Context* context) :
    myNodeInfo(info), myContext(context), myStream(0),
    myInputSurface(0), myOutputSurface(0), myError(nullptr)
{
    cudaStreamCreate(&myStream);
}

PixelStretchTOP::~PixelStretchTOP()
{
    if (myInputSurface)  cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface) cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)        cudaStreamDestroy(myStream);
}

void
PixelStretchTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
    ginfo->cookEveryFrame = false;
    ginfo->cookEveryFrameIfAsked = false;
    // Keep the Version field read-only here too (runs even when no input is connected
    // and execute would early-return).
    if (inputs) inputs->enablePar("Version", false);
}

void
PixelStretchTOP::getInfoPopupString(OP_String* info, void*)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "Pixel Stretch v%s", kVersion);
    info->setString(buf);
}

void
PixelStretchTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    myError = nullptr;
    inputs->enablePar("Version", false);   // make the Version field read-only

    if (inputs->getNumInputs() < 1)
    {
        myError = "Connect a TOP to the input.";
        return;
    }

    const OP_TOPInput* topInput = inputs->getInputTOP(0);
    if (!topInput)
    {
        myError = "Input TOP is invalid.";
        return;
    }

    const OP_TextureDesc& inDesc = topInput->textureDesc;

    if (inDesc.texDim != OP_TexDim::e2D)
    {
        myError = "Only 2D textures are supported (no 3D / cube / 2D-array).";
        return;
    }
    if (!isSupported8BitRGBA(inDesc.pixelFormat))
    {
        myError = "Input must be 8-bit RGBA/BGRA (BGRA8Fixed or RGBA8Fixed).";
        return;
    }
    if ((int)inDesc.width > pixelstretch::kMaxDim || (int)inDesc.height > pixelstretch::kMaxDim)
    {
        myError = "Resolution exceeds the 16384-pixel limit.";
        return;
    }

    TOP_CUDAOutputInfo info;
    info.textureDesc = inDesc;
    info.stream      = myStream;

    // All input/parameter queries must happen BEFORE beginCUDAOperations().
    OP_CUDAAcquireInfo acquireInfo;
    acquireInfo.stream = myStream;
    const OP_CUDAArrayInfo* inputArrayInfo = topInput->getCUDAArray(acquireInfo, nullptr);

    const OP_CUDAArrayInfo* outputArrayInfo = output->createCUDAArray(info, nullptr);
    if (!outputArrayInfo)
    {
        myError = "Failed to create output CUDA array.";
        return;
    }

    pixelstretch::Params p;
    p.width     = (int)inDesc.width;
    p.height    = (int)inDesc.height;
    p.bgra      = (inDesc.pixelFormat == OP_PixelFormat::BGRA8Fixed);
    p.axis      = (pixelstretch::Axis)   std::clamp(inputs->getParInt("Axis"), 0, 1);
    p.order     = (pixelstretch::Order)  std::clamp(inputs->getParInt("Order"), 0, 1);
    p.criterion = (pixelstretch::Channel)std::clamp(inputs->getParInt("Threshcrit"), 0, 7);
    p.threshold = (float)std::clamp(inputs->getParDouble("Threshold"), 0.0, 1.0);
    p.stretchLength = (float)std::clamp(inputs->getParDouble("Stretchlength"), 0.0, 1.0);
    p.fade       = inputs->getParInt("Fade") != 0;
    p.fadeAmount = (float)std::clamp(inputs->getParDouble("Fadeamount"), 0.0, 1.0);
    p.bypass     = inputs->getParInt("Bypass") != 0;

    if (!myContext->beginCUDAOperations(nullptr))
    {
        myError = "beginCUDAOperations() failed.";
        return;
    }

    setupCudaSurface(&myOutputSurface, outputArrayInfo->cudaArray);
    if (inputArrayInfo && inputArrayInfo->cudaArray)
    {
        setupCudaSurface(&myInputSurface, inputArrayInfo->cudaArray);
    }
    else if (myInputSurface)
    {
        cudaDestroySurfaceObject(myInputSurface);
        myInputSurface = 0;
    }

    // Swallow any benign sticky error from (re)creating surfaces (e.g. after a bypass
    // toggle), so the launch checks inside process() only report this cook's kernels.
    cudaGetLastError();

    const char* algoError = nullptr;
    myStretcher.process(myInputSurface, myOutputSurface, p, myStream, &algoError);
    if (algoError)
        myError = algoError;

    myContext->endCUDAOperations(nullptr);
}

void
PixelStretchTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError);
}

void
PixelStretchTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    const char* page = "Stretch";

    // Bypass — pass the input through untouched (the C++ TOP API does not report the
    // node's native Bypass flag to the plugin, so we expose our own that actually works).
    {
        OP_NumericParameter np("Bypass");
        np.label = "Bypass";
        np.page  = page;
        np.defaultValues[0] = 0.0;
        OP_ParAppendResult res = manager->appendToggle(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Stretch Axis
    {
        OP_StringParameter sp("Axis");
        sp.label = "Stretch Axis";
        sp.page  = page;
        sp.defaultValue = "Horizontal";
        const char* names[]  = { "Horizontal", "Vertical" };
        const char* labels[] = { "Horizontal", "Vertical" };
        OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }
    // Stretch Order — fill direction along the axis.
    {
        OP_StringParameter sp("Order");
        sp.label = "Stretch Order";
        sp.page  = page;
        sp.defaultValue = "Ascending";
        const char* names[]  = { "Ascending", "Descending" };
        const char* labels[] = { "Ascending", "Descending" };
        OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }
    // Threshold Criterion — what each pixel is thresholded on (default Luminance,
    // matching TD's Threshold TOP). Alpha reproduces the original alpha-based mode.
    {
        OP_StringParameter sp("Threshcrit");
        sp.label = "Threshold Criterion";
        sp.page  = page;
        sp.defaultValue = "Luminance";
        const char* names[]  = { "Luminance", "Hue", "Saturation", "Value",
                                 "Red", "Green", "Blue", "Alpha" };
        OP_ParAppendResult res = manager->appendMenu(sp, 8, names, names);
        assert(res == OP_ParAppendResult::Success);
    }
    // Threshold (1 = no effect, 0 = full effect / blank)
    {
        OP_NumericParameter np("Threshold");
        np.label = "Threshold";
        np.page  = page;
        np.defaultValues[0] = 0.5;
        np.minValues[0] = 0.0;  np.maxValues[0] = 1.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 1.0;
        np.clampMins[0] = true; np.clampMaxes[0] = true;
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Stretch Length — how far the smear travels into each held section (0..1).
    {
        OP_NumericParameter np("Stretchlength");
        np.label = "Stretch Length";
        np.page  = page;
        np.defaultValues[0] = 1.0;
        np.minValues[0] = 0.0;  np.maxValues[0] = 1.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 1.0;
        np.clampMins[0] = true; np.clampMaxes[0] = true;
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Fade to Black — ramp each held section from the survivor color to black.
    {
        OP_NumericParameter np("Fade");
        np.label = "Fade To Black";
        np.page  = page;
        np.defaultValues[0] = 1.0;
        OP_ParAppendResult res = manager->appendToggle(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Fade Amount — 0 = flat hold, 1 = reaches black at the end of each held section.
    {
        OP_NumericParameter np("Fadeamount");
        np.label = "Fade Amount";
        np.page  = page;
        np.defaultValues[0] = 1.0;
        np.minValues[0] = 0.0;  np.maxValues[0] = 1.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 1.0;
        np.clampMins[0] = true; np.clampMaxes[0] = true;
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }
    // Version — read-only string on its own tab (shows the version the node was made with).
    {
        OP_StringParameter sp("Version");
        sp.label = "Version";
        sp.page  = "Version";
        sp.defaultValue = kVersion;
        OP_ParAppendResult res = manager->appendString(sp);
        assert(res == OP_ParAppendResult::Success);
    }
}
