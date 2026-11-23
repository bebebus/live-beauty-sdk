#include "gles_device.h"

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <cstdio>

#include "gles_shaders.h"

namespace livebeauty::rhi::gles {

namespace {

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    if (!s) return 0;
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint linkProgram(const char* vSrc, const char* fSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vSrc);
    if (!vs) return 0;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fSrc);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

}  // namespace

GlesDevice::GlesDevice() {
    for (auto& pc : programs_) pc.uniLoc.fill(-1);
}

GlesDevice::~GlesDevice() {
    releaseAll(false);
}

void GlesDevice::releaseAll(bool contextAlive) {
    if (contextAlive) {
        for (auto& [id, rt] : rts_) {
            if (rt.fbo) glDeleteFramebuffers(1, &rt.fbo);
        }
        for (auto& [id, tex] : textures_) {
            if (!tex.external && tex.glTex) glDeleteTextures(1, &tex.glTex);
        }
        for (auto& [id, buf] : buffers_) {
            if (buf.glBuf) glDeleteBuffers(1, &buf.glBuf);
        }
        if (quadVbo_) glDeleteBuffers(1, &quadVbo_);
        if (oesProgram_) glDeleteProgram(oesProgram_);
        for (auto& pc : programs_) {
            if (pc.program) glDeleteProgram(pc.program);
        }
    }
    rts_.clear();
    textures_.clear();
    buffers_.clear();
    quadVbo_ = 0;
    oesProgram_ = 0;
    oesAttrPos_ = oesAttrUv_ = oesUniTex_ = oesUniMat_ = -1;
    oesRt_ = kInvalidRenderTarget;
    oesW_ = oesH_ = 0;
    for (auto& pc : programs_) {
        pc = ProgramCache{};
        pc.uniLoc.fill(-1);
    }
}

TextureId GlesDevice::wrapExternalTexture(uint32_t glTexId) {
    if (glTexId == 0) return kInvalidTexture;
    for (const auto& [id, rec] : textures_) {
        if (rec.external && rec.glTex == glTexId) return id;
    }
    const TextureId id = nextId_++;
    textures_[id] = TexRecord{glTexId, true};
    return id;
}

uint32_t GlesDevice::unwrapGlTexture(TextureId tex) const {
    auto it = textures_.find(tex);
    return (it != textures_.end()) ? it->second.glTex : 0u;
}

TextureId GlesDevice::convertOesTo2D(uint32_t oesTexId, int width, int height, const float* stMatrix16) {
    if (oesTexId == 0 || width <= 0 || height <= 0) return kInvalidTexture;
    if (oesProgram_ == 0) {
        static const char* kOesVert = R"(
attribute vec4 aPosition;
attribute vec4 aTextureCoord;
uniform mat4 uTexMatrix;
varying vec2 vTextureCoord;
void main() {
    gl_Position = aPosition;
    vTextureCoord = (uTexMatrix * aTextureCoord).xy;
}
)";
        static const char* kOesFrag = R"(
#extension GL_OES_EGL_image_external : require
precision mediump float;
varying vec2 vTextureCoord;
uniform samplerExternalOES uTex0;
void main() {
    gl_FragColor = texture2D(uTex0, vTextureCoord);
}
)";
        oesProgram_ = linkProgram(kOesVert, kOesFrag);
        if (oesProgram_ == 0) return kInvalidTexture;
        oesAttrPos_ = glGetAttribLocation(oesProgram_, "aPosition");
        oesAttrUv_ = glGetAttribLocation(oesProgram_, "aTextureCoord");
        oesUniTex_ = glGetUniformLocation(oesProgram_, "uTex0");
        oesUniMat_ = glGetUniformLocation(oesProgram_, "uTexMatrix");
    }
    if (oesRt_ == kInvalidRenderTarget || oesW_ != width || oesH_ != height) {
        if (oesRt_ != kInvalidRenderTarget) destroyRenderTarget(oesRt_);
        oesRt_ = createRenderTarget(width, height, TextureFormat::kRgba8);
        oesW_ = width;
        oesH_ = height;
    }
    static constexpr float kIdentity16[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const float* mat = stMatrix16 ? stMatrix16 : kIdentity16;
    ensureQuadVbo();
    beginPass(oesRt_, width, height, false, {});
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glUseProgram(oesProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, oesTexId);
    if (oesUniTex_ >= 0) glUniform1i(oesUniTex_, 0);
    if (oesUniMat_ >= 0) glUniformMatrix4fv(oesUniMat_, 1, GL_FALSE, mat);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo_);
    if (oesAttrPos_ >= 0) {
        glEnableVertexAttribArray(oesAttrPos_);
        glVertexAttribPointer(oesAttrPos_, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<const void*>(0));
    }
    if (oesAttrUv_ >= 0) {
        glEnableVertexAttribArray(oesAttrUv_);
        glVertexAttribPointer(oesAttrUv_, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              reinterpret_cast<const void*>(2 * sizeof(float)));
    }
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    if (oesAttrPos_ >= 0) glDisableVertexAttribArray(oesAttrPos_);
    if (oesAttrUv_ >= 0) glDisableVertexAttribArray(oesAttrUv_);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);
    restoreSharedState();
    endPass();
    return textureOf(oesRt_);
}

RenderTargetId GlesDevice::createRenderTarget(int width, int height, TextureFormat format) {
    const TextureId texId = createTexture2D(width, height, format, nullptr);
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, unwrapGlTexture(texId), 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    const RenderTargetId rtId = nextId_++;
    rts_[rtId] = RtRecord{fbo, texId, width, height};
    return rtId;
}

void GlesDevice::destroyRenderTarget(RenderTargetId rt) {
    auto it = rts_.find(rt);
    if (it == rts_.end()) return;
    if (it->second.fbo) glDeleteFramebuffers(1, &it->second.fbo);
    destroyTexture(it->second.texId);
    rts_.erase(it);
}

TextureId GlesDevice::textureOf(RenderTargetId rt) const {
    auto it = rts_.find(rt);
    return (it != rts_.end()) ? it->second.texId : kInvalidTexture;
}

TextureId GlesDevice::createTexture2D(int width, int height, TextureFormat format, const uint8_t* pixels) {
    GLuint glTex = 0;
    glGenTextures(1, &glTex);
    glBindTexture(GL_TEXTURE_2D, glTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const GLenum glFmt = (format == TextureFormat::kR8) ? GL_LUMINANCE : GL_RGBA;
    glPixelStorei(GL_UNPACK_ALIGNMENT, (format == TextureFormat::kR8) ? 1 : 4);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(glFmt), width, height, 0, glFmt, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
    const TextureId id = nextId_++;
    textures_[id] = TexRecord{glTex, false};
    return id;
}

void GlesDevice::updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) {
    const uint32_t glTex = unwrapGlTexture(tex);
    if (!glTex || !pixels) return;
    glBindTexture(GL_TEXTURE_2D, glTex);
    const GLenum glFmt = (format == TextureFormat::kR8) ? GL_LUMINANCE : GL_RGBA;
    glPixelStorei(GL_UNPACK_ALIGNMENT, (format == TextureFormat::kR8) ? 1 : 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, glFmt, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GlesDevice::destroyTexture(TextureId tex) {
    auto it = textures_.find(tex);
    if (it == textures_.end()) return;
    if (!it->second.external && it->second.glTex) glDeleteTextures(1, &it->second.glTex);
    textures_.erase(it);
}

BufferId GlesDevice::createDynamicVertexBuffer(size_t byteSize) {
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(byteSize), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    const BufferId id = nextId_++;
    buffers_[id] = BufRecord{vbo, false};
    return id;
}

void GlesDevice::updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) {
    auto it = buffers_.find(buf);
    if (it == buffers_.end() || !data) return;
    glBindBuffer(GL_ARRAY_BUFFER, it->second.glBuf);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(byteSize), data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

BufferId GlesDevice::createStaticIndexBuffer(const uint16_t* indices, size_t count) {
    GLuint ibo = 0;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(count * sizeof(uint16_t)), indices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    const BufferId id = nextId_++;
    buffers_[id] = BufRecord{ibo, true};
    return id;
}

void GlesDevice::destroyBuffer(BufferId buf) {
    auto it = buffers_.find(buf);
    if (it == buffers_.end()) return;
    if (it->second.glBuf) glDeleteBuffers(1, &it->second.glBuf);
    buffers_.erase(it);
}

void GlesDevice::ensureQuadVbo() {
    if (quadVbo_ != 0) return;
    static constexpr float kQuad[16] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
    };
    glGenBuffers(1, &quadVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuad), kQuad, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

const GlesDevice::ProgramCache& GlesDevice::ensureProgram(ShaderId id) {
    auto& pc = programs_[static_cast<size_t>(id)];
    if (pc.program != 0) return pc;
    const auto pair = getShaderSources(id);
    pc.program = linkProgram(pair.vertexSrc, pair.fragmentSrc);
    if (pc.program == 0) return pc;
    pc.attr0 = glGetAttribLocation(pc.program, "aPosition");
    if (id == ShaderId::kSkinMask) {
        pc.attr1 = glGetAttribLocation(pc.program, "aMaskWeights");
    } else if (id == ShaderId::kRetouchSkin || id == ShaderId::kRetouchEyeTeeth) {
        pc.attr1 = glGetAttribLocation(pc.program, "aSrcCoord");
        pc.attr2 = glGetAttribLocation(pc.program, "aRetouchAttr");
    } else if (id == ShaderId::kMakeupLayer || id == ShaderId::kMakeupLip || id == ShaderId::kMakeupIris) {
        pc.attr1 = glGetAttribLocation(pc.program, "aScreenUv");
        pc.attr2 = glGetAttribLocation(pc.program, "aMakeupUv");
    } else {
        pc.attr1 = glGetAttribLocation(pc.program, "aTextureCoord");
    }
    char name[16];
    for (int i = 0; i < 4; ++i) {
        std::snprintf(name, sizeof(name), "uTex%d", i);
        pc.texLoc[i] = glGetUniformLocation(pc.program, name);
    }
    for (int i = 0; i < 16; ++i) {
        std::snprintf(name, sizeof(name), "uU%d", i);
        pc.uniLoc[i] = glGetUniformLocation(pc.program, name);
    }
    return pc;
}

void GlesDevice::beginPass(RenderTargetId rt, int viewportW, int viewportH, bool clear, const Vec4& clearColor) {
    auto it = rts_.find(rt);
    glBindFramebuffer(GL_FRAMEBUFFER, (it != rts_.end()) ? it->second.fbo : 0);
    glViewport(0, 0, viewportW, viewportH);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    if (clear) {
        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

void GlesDevice::applyStateAndUniforms(
    const ProgramCache& pc, const RenderState& state,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    if (state.blendEnable) {
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    } else {
        glDisable(GL_BLEND);
    }
    if (state.cullBack) {
        glEnable(GL_CULL_FACE);
        glFrontFace(state.frontFaceCw ? GL_CW : GL_CCW);
        glCullFace(GL_BACK);
    } else {
        glDisable(GL_CULL_FACE);
        glFrontFace(GL_CCW);
    }
    glColorMask(!state.colorMaskAlphaOnly, !state.colorMaskAlphaOnly, !state.colorMaskAlphaOnly, GL_TRUE);

    glUseProgram(pc.program);
    for (int i = 0; i < textures.count; ++i) {
        if (pc.texLoc[i] >= 0 && textures.slots[i] != kInvalidTexture) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, unwrapGlTexture(textures.slots[i]));
            glUniform1i(pc.texLoc[i], i);
        }
    }
    for (int i = 0; i < uniforms.count; ++i) {
        if (pc.uniLoc[i] >= 0) {
            const auto& v = uniforms.v[i];
            glUniform4f(pc.uniLoc[i], v.x, v.y, v.z, v.w);
        }
    }
}

void GlesDevice::restoreSharedState() {
    glDisable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    // Unbind all 4 sampler units so no FBO color attachment stays bound on another unit in subsequent passes.
    for (int i = 3; i >= 0; --i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void GlesDevice::drawFullScreenQuad(
    ShaderId shader, const RenderState& state,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    const auto& pc = ensureProgram(shader);
    if (!pc.program) return;
    ensureQuadVbo();
    applyStateAndUniforms(pc, state, textures, uniforms);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo_);
    if (pc.attr0 >= 0) {
        glEnableVertexAttribArray(pc.attr0);
        glVertexAttribPointer(pc.attr0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<const void*>(0));
    }
    if (pc.attr1 >= 0) {
        glEnableVertexAttribArray(pc.attr1);
        glVertexAttribPointer(pc.attr1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              reinterpret_cast<const void*>(2 * sizeof(float)));
    }
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    if (pc.attr0 >= 0) glDisableVertexAttribArray(pc.attr0);
    if (pc.attr1 >= 0) glDisableVertexAttribArray(pc.attr1);
    restoreSharedState();
}

void GlesDevice::drawIndexed(
    ShaderId shader, const RenderState& state,
    BufferId vertexBuffer, BufferId indexBuffer,
    int indexOffset, int indexCount, VertexLayout layout,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    const auto& pc = ensureProgram(shader);
    auto vbIt = buffers_.find(vertexBuffer);
    auto ibIt = buffers_.find(indexBuffer);
    if (!pc.program || vbIt == buffers_.end() || ibIt == buffers_.end()) return;

    applyStateAndUniforms(pc, state, textures, uniforms);
    glBindBuffer(GL_ARRAY_BUFFER, vbIt->second.glBuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibIt->second.glBuf);

    auto bindAttr = [](int loc, int comps, int strideFloats, int offsetFloats) {
        if (loc >= 0) {
            glEnableVertexAttribArray(loc);
            glVertexAttribPointer(
                loc, comps, GL_FLOAT, GL_FALSE,
                strideFloats * static_cast<int>(sizeof(float)),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(offsetFloats) * sizeof(float)));
        }
    };
    switch (layout) {
        case VertexLayout::kMeshPos2Uv2:
        case VertexLayout::kQuadPos2Uv2:
            bindAttr(pc.attr0, 2, 4, 0);
            bindAttr(pc.attr1, 2, 4, 2);
            break;
        case VertexLayout::kSkinMaskPos2W3:
            bindAttr(pc.attr0, 2, 5, 0);
            bindAttr(pc.attr1, 3, 5, 2);
            break;
        case VertexLayout::kRetouchPos2Uv2A4:
            bindAttr(pc.attr0, 2, 8, 0);
            bindAttr(pc.attr1, 2, 8, 2);
            bindAttr(pc.attr2, 4, 8, 4);
            break;
        case VertexLayout::kMakeupPos2Uv2Uv2:
            bindAttr(pc.attr0, 2, 6, 0);
            bindAttr(pc.attr1, 2, 6, 2);
            bindAttr(pc.attr2, 2, 6, 4);
            break;
    }

    glDrawElements(
        GL_TRIANGLES, indexCount, GL_UNSIGNED_SHORT,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(indexOffset) * sizeof(uint16_t)));

    if (pc.attr0 >= 0) glDisableVertexAttribArray(pc.attr0);
    if (pc.attr1 >= 0) glDisableVertexAttribArray(pc.attr1);
    if (pc.attr2 >= 0) glDisableVertexAttribArray(pc.attr2);
    restoreSharedState();
}

void GlesDevice::endPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}  // namespace livebeauty::rhi::gles
