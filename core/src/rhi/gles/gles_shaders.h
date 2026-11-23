#pragma once

#include "livebeauty/rhi/rhi_types.h"

namespace livebeauty::rhi::gles {

struct ShaderSourcePair {
    const char* vertexSrc;
    const char* fragmentSrc;
};

/** Returns GLSL ES 1.00 vertex/fragment source pair matching VideoEffectsShaders.kt formulas. */
ShaderSourcePair getShaderSources(ShaderId id);

}  // namespace livebeauty::rhi::gles
