/* Pixel Stretch TOP — TD SDK glue.
 * Validated: TouchDesigner 2025.32050, TOP C++ API v12.
 */
#ifndef PIXELSTRETCH_TOP_H
#define PIXELSTRETCH_TOP_H

#include "TOP_CPlusPlusBase.h"
#include "cuda_runtime.h"
#include "PixelStretchCUDA.h"

using namespace TD;

class PixelStretchTOP : public TOP_CPlusPlusBase
{
public:
    PixelStretchTOP(const OP_NodeInfo* info, TOP_Context* context);
    virtual ~PixelStretchTOP();

    virtual void    getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;
    virtual void    execute(TOP_Output*, const OP_Inputs*, void* reserved1) override;

    virtual void    getErrorString(OP_String* error, void* reserved1) override;
    virtual void    getInfoPopupString(OP_String* info, void* reserved1) override;

    virtual void    setupParameters(OP_ParameterManager* manager, void* reserved1) override;

private:
    const OP_NodeInfo*  myNodeInfo;
    TOP_Context*        myContext;
    cudaStream_t        myStream;

    cudaSurfaceObject_t myInputSurface;
    cudaSurfaceObject_t myOutputSurface;

    pixelstretch::PixelStretcher  myStretcher;

    const char*         myError;
};

#endif // PIXELSTRETCH_TOP_H
