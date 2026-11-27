#include "livebeauty/rhi/webgl/webgl2_device.h"

#include <cstdio>

#include "../gles/gles_shaders.h"

#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#include <emscripten/html5_webgl.h>
#endif

namespace livebeauty::rhi::webgl {

namespace {

// Standard OpenGL ES 3.0 / WebGL 2.0 token constants
constexpr uint32_t kGlNoError              = 0x0000;
constexpr uint32_t kGlTriangles            = 0x0004;
constexpr uint32_t kGlTriangleStrip        = 0x0005;
constexpr uint32_t kGlZero                 = 0x0000;
constexpr uint32_t kGlOne                  = 0x0001;
constexpr uint32_t kGlSrcAlpha             = 0x0302;
constexpr uint32_t kGlOneMinusSrcAlpha     = 0x0303;
constexpr uint32_t kGlCullFace             = 0x0B44;
constexpr uint32_t kGlDepthTest            = 0x0B71;
constexpr uint32_t kGlBlend                = 0x0BE2;
constexpr uint32_t kGlScissorTest          = 0x0C11;
constexpr uint32_t kGlUnpackAlignment      = 0x0CF5;
constexpr uint32_t kGlTexture2D            = 0x0DE1;
constexpr uint32_t kGlUnsignedByte         = 0x1401;
constexpr uint32_t kGlUnsignedShort        = 0x1403;
constexpr uint32_t kGlFloat                = 0x1406;
constexpr uint32_t kGlHalfFloat            = 0x140B;
constexpr uint32_t kGlRed                  = 0x1903;
constexpr uint32_t kGlRgba                 = 0x1908;
constexpr uint32_t kGlCw                   = 0x0900;
constexpr uint32_t kGlCcw                  = 0x0901;
constexpr uint32_t kGlBack                 = 0x0405;
constexpr uint32_t kGlLinear               = 0x2601;
constexpr uint32_t kGlTextureMagFilter     = 0x2800;
constexpr uint32_t kGlTextureMinFilter     = 0x2801;
constexpr uint32_t kGlTextureWrapS         = 0x2802;
constexpr uint32_t kGlTextureWrapT         = 0x2803;
constexpr uint32_t kGlClampToEdge          = 0x812F;
constexpr uint32_t kGlColorBufferBit       = 0x4000;
constexpr uint32_t kGlRgba8                = 0x8058;
constexpr uint32_t kGlR8                   = 0x8229;
constexpr uint32_t kGlTexture0             = 0x84C0;
constexpr uint32_t kGlRgba16f              = 0x881A;
constexpr uint32_t kGlArrayBuffer          = 0x8892;
constexpr uint32_t kGlElementArrayBuffer   = 0x8893;
constexpr uint32_t kGlStaticDraw           = 0x88E4;
constexpr uint32_t kGlDynamicDraw          = 0x88E8;
constexpr uint32_t kGlFragmentShader       = 0x8B30;
constexpr uint32_t kGlVertexShader         = 0x8B31;
constexpr uint32_t kGlCompileStatus        = 0x8B81;
constexpr uint32_t kGlLinkStatus           = 0x8B82;
constexpr uint32_t kGlFramebuffer          = 0x8D40;
constexpr uint32_t kGlColorAttachment0     = 0x8CE0;
constexpr uint32_t kGlFramebufferComplete  = 0x8CD5;

}  // namespace

#ifdef __EMSCRIPTEN__
WebGl2DispatchTable WebGl2DispatchTable::emscriptenDefault() {
    WebGl2DispatchTable t{};
    t.genTextures = [](int n, uint32_t* ids) { glGenTextures(n, ids); };
    t.deleteTextures = [](int n, const uint32_t* ids) { glDeleteTextures(n, ids); };
    t.bindTexture = [](uint32_t target, uint32_t tex) { glBindTexture(target, tex); };
    t.activeTexture = [](uint32_t unit) { glActiveTexture(unit); };
    t.texParameteri = [](uint32_t target, uint32_t pname, int param) { glTexParameteri(target, pname, param); };
    t.pixelStorei = [](uint32_t pname, int param) { glPixelStorei(pname, param); };
    t.texImage2D = [](uint32_t target, int level, int internalFmt, int w, int h, int border,
                      uint32_t fmt, uint32_t type, const void* px) {
        glTexImage2D(target, level, internalFmt, w, h, border, fmt, type, px);
    };
    t.texSubImage2D = [](uint32_t target, int level, int x, int y, int w, int h,
                         uint32_t fmt, uint32_t type, const void* px) {
        glTexSubImage2D(target, level, x, y, w, h, fmt, type, px);
    };
    t.genFramebuffers = [](int n, uint32_t* ids) { glGenFramebuffers(n, ids); };
    t.deleteFramebuffers = [](int n, const uint32_t* ids) { glDeleteFramebuffers(n, ids); };
    t.bindFramebuffer = [](uint32_t target, uint32_t fbo) { glBindFramebuffer(target, fbo); };
    t.framebufferTexture2D = [](uint32_t target, uint32_t att, uint32_t texTarget, uint32_t tex, int level) {
        glFramebufferTexture2D(target, att, texTarget, tex, level);
    };
    t.checkFramebufferStatus = [](uint32_t target) -> uint32_t { return glCheckFramebufferStatus(target); };
    t.genBuffers = [](int n, uint32_t* ids) { glGenBuffers(n, ids); };
    t.deleteBuffers = [](int n, const uint32_t* ids) { glDeleteBuffers(n, ids); };
    t.bindBuffer = [](uint32_t target, uint32_t buf) { glBindBuffer(target, buf); };
    t.bufferData = [](uint32_t target, ptrdiff_t size, const void* data, uint32_t usage) {
        glBufferData(target, size, data, usage);
    };
    t.createShader = [](uint32_t type) -> uint32_t { return glCreateShader(type); };
    t.deleteShader = [](uint32_t s) { glDeleteShader(s); };
    t.shaderSource = [](uint32_t s, int count, const char* const* str, const int* len) {
        glShaderSource(s, count, str, len);
    };
    t.compileShader = [](uint32_t s) { glCompileShader(s); };
    t.getShaderiv = [](uint32_t s, uint32_t pname, int* out) { glGetShaderiv(s, pname, out); };
    t.createProgram = []() -> uint32_t { return glCreateProgram(); };
    t.deleteProgram = [](uint32_t p) { glDeleteProgram(p); };
    t.attachShader = [](uint32_t p, uint32_t s) { glAttachShader(p, s); };
    t.linkProgram = [](uint32_t p) { glLinkProgram(p); };
    t.getProgramiv = [](uint32_t p, uint32_t pname, int* out) { glGetProgramiv(p, pname, out); };
    t.useProgram = [](uint32_t p) { glUseProgram(p); };
    t.getAttribLocation = [](uint32_t p, const char* name) -> int { return glGetAttribLocation(p, name); };
    t.getUniformLocation = [](uint32_t p, const char* name) -> int { return glGetUniformLocation(p, name); };
    t.enableVertexAttribArray = [](uint32_t idx) { glEnableVertexAttribArray(idx); };
    t.disableVertexAttribArray = [](uint32_t idx) { glDisableVertexAttribArray(idx); };
    t.vertexAttribPointer = [](uint32_t idx, int size, uint32_t type, uint8_t norm, int stride, const void* off) {
        glVertexAttribPointer(idx, size, type, norm, stride, off);
    };
    t.uniform1i = [](int loc, int v0) { glUniform1i(loc, v0); };
    t.uniform4f = [](int loc, float x, float y, float z, float w) { glUniform4f(loc, x, y, z, w); };
    t.viewport = [](int x, int y, int w, int h) { glViewport(x, y, w, h); };
    t.enable = [](uint32_t cap) { glEnable(cap); };
    t.disable = [](uint32_t cap) { glDisable(cap); };
    t.blendFuncSeparate = [](uint32_t sr, uint32_t dr, uint32_t sa, uint32_t da) {
        glBlendFuncSeparate(sr, dr, sa, da);
    };
    t.frontFace = [](uint32_t m) { glFrontFace(m); };
    t.cullFace = [](uint32_t m) { glCullFace(m); };
    t.colorMask = [](uint8_t r, uint8_t g, uint8_t b, uint8_t a) { glColorMask(r, g, b, a); };
    t.clearColor = [](float r, float g, float b, float a) { glClearColor(r, g, b, a); };
    t.clear = [](uint32_t mask) { glClear(mask); };
    t.drawArrays = [](uint32_t mode, int first, int count) { glDrawArrays(mode, first, count); };
    t.drawElements = [](uint32_t mode, int count, uint32_t type, const void* idx) {
        glDrawElements(mode, count, type, idx);
    };
    t.getError = []() -> uint32_t { return glGetError(); };
    t.enableExtension = [](const char* ext) -> bool {
        const EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_get_current_context();
        return ctx > 0 && emscripten_webgl_enable_extension(ctx, ext) == EM_TRUE;
    };
    return t;
}

WebGl2Device::WebGl2Device() : WebGl2Device(WebGl2DispatchTable::emscriptenDefault()) {}
#endif

WebGl2Device::WebGl2Device(const WebGl2DispatchTable& gl) : gl_(gl) {
    for (auto& pc : programs_) pc.uniLoc.fill(-1);
    if (gl_.enableExtension) {
        const bool cbf = gl_.enableExtension("EXT_color_buffer_float") ||
                         gl_.enableExtension("EXT_color_buffer_half_float");
        gl_.enableExtension("OES_texture_float_linear");
        colorBufferFloatExt_ = cbf;
    }
}

WebGl2Device::~WebGl2Device() {
    releaseAll(false);
}

bool WebGl2Device::isReady() const {
    return gl_.genTextures && gl_.bindTexture && gl_.texParameteri && gl_.pixelStorei &&
           gl_.texImage2D && gl_.texSubImage2D && gl_.genFramebuffers && gl_.bindFramebuffer &&
           gl_.framebufferTexture2D && gl_.checkFramebufferStatus && gl_.genBuffers &&
           gl_.bindBuffer && gl_.bufferData && gl_.createShader && gl_.deleteShader &&
           gl_.shaderSource && gl_.compileShader && gl_.getShaderiv && gl_.createProgram &&
           gl_.deleteProgram && gl_.attachShader && gl_.linkProgram && gl_.getProgramiv &&
           gl_.useProgram && gl_.getAttribLocation && gl_.getUniformLocation &&
           gl_.enableVertexAttribArray && gl_.disableVertexAttribArray && gl_.vertexAttribPointer &&
           gl_.uniform1i && gl_.uniform4f && gl_.viewport && gl_.enable && gl_.disable &&
           gl_.blendFuncSeparate && gl_.frontFace && gl_.cullFace && gl_.colorMask &&
           gl_.clearColor && gl_.clear && gl_.drawArrays && gl_.drawElements &&
           gl_.activeTexture;
}

void WebGl2Device::releaseAll(bool contextAlive) {
    if (contextAlive) {
        for (auto& [id, rt] : rts_) {
            if (rt.fbo && gl_.deleteFramebuffers) gl_.deleteFramebuffers(1, &rt.fbo);
        }
        for (auto& [id, tex] : textures_) {
            if (!tex.external && tex.glTex && gl_.deleteTextures) gl_.deleteTextures(1, &tex.glTex);
        }
        for (auto& [id, buf] : buffers_) {
            if (buf.glBuf && gl_.deleteBuffers) gl_.deleteBuffers(1, &buf.glBuf);
        }
        if (quadVbo_ && gl_.deleteBuffers) gl_.deleteBuffers(1, &quadVbo_);
        for (auto& pc : programs_) {
            if (pc.program && gl_.deleteProgram) gl_.deleteProgram(pc.program);
        }
    }
    rts_.clear();
    textures_.clear();
    buffers_.clear();
    quadVbo_ = 0;
    for (auto& pc : programs_) {
        pc = ProgramCache{};
        pc.uniLoc.fill(-1);
    }
}

TextureId WebGl2Device::wrapExternalTexture(uint32_t webglTexId) {
    if (webglTexId == 0) return kInvalidTexture;
    for (const auto& [id, rec] : textures_) {
        if (rec.external && rec.glTex == webglTexId) return id;
    }
    const TextureId id = nextId_++;
    textures_[id] = TexRecord{webglTexId, true};
    return id;
}

uint32_t WebGl2Device::unwrapWebGlTexture(TextureId tex) const {
    auto it = textures_.find(tex);
    return (it != textures_.end()) ? it->second.glTex : 0u;
}

uint32_t WebGl2Device::compileShader(uint32_t type, const char* src) {
    if (!gl_.createShader) return 0;
    const uint32_t s = gl_.createShader(type);
    if (!s) return 0;
    gl_.shaderSource(s, 1, &src, nullptr);
    gl_.compileShader(s);
    int ok = 0;
    gl_.getShaderiv(s, kGlCompileStatus, &ok);
    if (!ok) {
        gl_.deleteShader(s);
        return 0;
    }
    return s;
}

uint32_t WebGl2Device::linkProgram(const char* vSrc, const char* fSrc) {
    const uint32_t vs = compileShader(kGlVertexShader, vSrc);
    if (!vs) return 0;
    const uint32_t fs = compileShader(kGlFragmentShader, fSrc);
    if (!fs) {
        gl_.deleteShader(vs);
        return 0;
    }
    const uint32_t p = gl_.createProgram();
    gl_.attachShader(p, vs);
    gl_.attachShader(p, fs);
    gl_.linkProgram(p);
    gl_.deleteShader(vs);
    gl_.deleteShader(fs);
    int ok = 0;
    gl_.getProgramiv(p, kGlLinkStatus, &ok);
    if (!ok) {
        gl_.deleteProgram(p);
        return 0;
    }
    return p;
}

RenderTargetId WebGl2Device::createRenderTarget(int width, int height, TextureFormat format) {
    if (!isReady() || width <= 0 || height <= 0) return kInvalidRenderTarget;
    if (format == TextureFormat::kRgba16f && colorBufferFloatExt_) {
        while (gl_.getError && gl_.getError() != kGlNoError) {}
        uint32_t glTex = 0;
        gl_.genTextures(1, &glTex);
        gl_.bindTexture(kGlTexture2D, glTex);
        gl_.texParameteri(kGlTexture2D, kGlTextureMinFilter, static_cast<int>(kGlLinear));
        gl_.texParameteri(kGlTexture2D, kGlTextureMagFilter, static_cast<int>(kGlLinear));
        gl_.texParameteri(kGlTexture2D, kGlTextureWrapS, static_cast<int>(kGlClampToEdge));
        gl_.texParameteri(kGlTexture2D, kGlTextureWrapT, static_cast<int>(kGlClampToEdge));
        gl_.texImage2D(kGlTexture2D, 0, static_cast<int>(kGlRgba16f), width, height, 0,
                       kGlRgba, kGlHalfFloat, nullptr);

        uint32_t fbo = 0;
        gl_.genFramebuffers(1, &fbo);
        gl_.bindFramebuffer(kGlFramebuffer, fbo);
        gl_.framebufferTexture2D(kGlFramebuffer, kGlColorAttachment0, kGlTexture2D, glTex, 0);
        const bool ok = (!gl_.getError || gl_.getError() == kGlNoError) &&
                        gl_.checkFramebufferStatus(kGlFramebuffer) == kGlFramebufferComplete;
        if (!ok) {
            while (gl_.getError && gl_.getError() != kGlNoError) {}
            gl_.texImage2D(kGlTexture2D, 0, static_cast<int>(kGlRgba8), width, height, 0,
                           kGlRgba, kGlUnsignedByte, nullptr);
            gl_.framebufferTexture2D(kGlFramebuffer, kGlColorAttachment0, kGlTexture2D, glTex, 0);
        }
        gl_.bindTexture(kGlTexture2D, 0);
        gl_.bindFramebuffer(kGlFramebuffer, 0);
        const TextureId texId = nextId_++;
        textures_[texId] = TexRecord{glTex, false};
        const RenderTargetId rtId = nextId_++;
        rts_[rtId] = RtRecord{fbo, texId, width, height};
        return rtId;
    }

    const TextureFormat fallbackFmt = (format == TextureFormat::kRgba16f) ? TextureFormat::kRgba8 : format;
    const TextureId texId = createTexture2D(width, height, fallbackFmt, nullptr);
    if (texId == kInvalidTexture) return kInvalidRenderTarget;
    uint32_t fbo = 0;
    gl_.genFramebuffers(1, &fbo);
    gl_.bindFramebuffer(kGlFramebuffer, fbo);
    gl_.framebufferTexture2D(kGlFramebuffer, kGlColorAttachment0, kGlTexture2D, unwrapWebGlTexture(texId), 0);
    gl_.bindFramebuffer(kGlFramebuffer, 0);
    const RenderTargetId rtId = nextId_++;
    rts_[rtId] = RtRecord{fbo, texId, width, height};
    return rtId;
}

void WebGl2Device::destroyRenderTarget(RenderTargetId rt) {
    auto it = rts_.find(rt);
    if (it == rts_.end()) return;
    if (it->second.fbo && gl_.deleteFramebuffers) gl_.deleteFramebuffers(1, &it->second.fbo);
    destroyTexture(it->second.texId);
    rts_.erase(it);
}

TextureId WebGl2Device::textureOf(RenderTargetId rt) const {
    auto it = rts_.find(rt);
    return (it != rts_.end()) ? it->second.texId : kInvalidTexture;
}

TextureId WebGl2Device::createTexture2D(int width, int height, TextureFormat format, const uint8_t* pixels) {
    if (!isReady() || width <= 0 || height <= 0) return kInvalidTexture;
    uint32_t glTex = 0;
    gl_.genTextures(1, &glTex);
    gl_.bindTexture(kGlTexture2D, glTex);
    gl_.texParameteri(kGlTexture2D, kGlTextureMinFilter, static_cast<int>(kGlLinear));
    gl_.texParameteri(kGlTexture2D, kGlTextureMagFilter, static_cast<int>(kGlLinear));
    gl_.texParameteri(kGlTexture2D, kGlTextureWrapS, static_cast<int>(kGlClampToEdge));
    gl_.texParameteri(kGlTexture2D, kGlTextureWrapT, static_cast<int>(kGlClampToEdge));
    // WebGL2 uses sized GL_R8 + GL_RED for single-channel masks (avoids deprecated GL_LUMINANCE).
    const int internalFmt = (format == TextureFormat::kR8)
        ? static_cast<int>(kGlR8)
        : (format == TextureFormat::kRgba16f ? static_cast<int>(kGlRgba16f) : static_cast<int>(kGlRgba8));
    const uint32_t baseFmt = (format == TextureFormat::kR8) ? kGlRed : kGlRgba;
    const uint32_t dataType = (format == TextureFormat::kRgba16f) ? kGlHalfFloat : kGlUnsignedByte;
    gl_.pixelStorei(kGlUnpackAlignment, (format == TextureFormat::kR8) ? 1 : 4);
    gl_.texImage2D(kGlTexture2D, 0, internalFmt, width, height, 0, baseFmt, dataType, pixels);
    gl_.bindTexture(kGlTexture2D, 0);
    const TextureId id = nextId_++;
    textures_[id] = TexRecord{glTex, false};
    return id;
}

void WebGl2Device::updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) {
    if (!isReady() || width <= 0 || height <= 0) return;
    const uint32_t glTex = unwrapWebGlTexture(tex);
    if (!glTex || !pixels) return;
    gl_.bindTexture(kGlTexture2D, glTex);
    const uint32_t baseFmt = (format == TextureFormat::kR8) ? kGlRed : kGlRgba;
    const uint32_t dataType = (format == TextureFormat::kRgba16f) ? kGlHalfFloat : kGlUnsignedByte;
    gl_.pixelStorei(kGlUnpackAlignment, (format == TextureFormat::kR8) ? 1 : 4);
    gl_.texSubImage2D(kGlTexture2D, 0, 0, 0, width, height, baseFmt, dataType, pixels);
    gl_.bindTexture(kGlTexture2D, 0);
}

void WebGl2Device::destroyTexture(TextureId tex) {
    auto it = textures_.find(tex);
    if (it == textures_.end()) return;
    if (!it->second.external && it->second.glTex && gl_.deleteTextures) {
        gl_.deleteTextures(1, &it->second.glTex);
    }
    textures_.erase(it);
}

BufferId WebGl2Device::createDynamicVertexBuffer(size_t byteSize) {
    if (!isReady() || byteSize == 0) return kInvalidBuffer;
    uint32_t vbo = 0;
    gl_.genBuffers(1, &vbo);
    gl_.bindBuffer(kGlArrayBuffer, vbo);
    gl_.bufferData(kGlArrayBuffer, static_cast<ptrdiff_t>(byteSize), nullptr, kGlDynamicDraw);
    gl_.bindBuffer(kGlArrayBuffer, 0);
    const BufferId id = nextId_++;
    buffers_[id] = BufRecord{vbo};
    return id;
}

void WebGl2Device::updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) {
    if (!isReady() || !data || byteSize == 0) return;
    auto it = buffers_.find(buf);
    if (it == buffers_.end()) return;
    gl_.bindBuffer(kGlArrayBuffer, it->second.glBuf);
    gl_.bufferData(kGlArrayBuffer, static_cast<ptrdiff_t>(byteSize), data, kGlDynamicDraw);
    gl_.bindBuffer(kGlArrayBuffer, 0);
}

BufferId WebGl2Device::createStaticIndexBuffer(const uint16_t* indices, size_t count) {
    if (!isReady() || !indices || count == 0) return kInvalidBuffer;
    uint32_t ibo = 0;
    gl_.genBuffers(1, &ibo);
    gl_.bindBuffer(kGlElementArrayBuffer, ibo);
    gl_.bufferData(kGlElementArrayBuffer, static_cast<ptrdiff_t>(count * sizeof(uint16_t)), indices, kGlStaticDraw);
    gl_.bindBuffer(kGlElementArrayBuffer, 0);
    const BufferId id = nextId_++;
    buffers_[id] = BufRecord{ibo};
    return id;
}

void WebGl2Device::destroyBuffer(BufferId buf) {
    auto it = buffers_.find(buf);
    if (it == buffers_.end()) return;
    if (it->second.glBuf && gl_.deleteBuffers) gl_.deleteBuffers(1, &it->second.glBuf);
    buffers_.erase(it);
}

void WebGl2Device::ensureQuadVbo() {
    if (quadVbo_ != 0) return;
    static constexpr float kQuad[16] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
    };
    gl_.genBuffers(1, &quadVbo_);
    gl_.bindBuffer(kGlArrayBuffer, quadVbo_);
    gl_.bufferData(kGlArrayBuffer, sizeof(kQuad), kQuad, kGlStaticDraw);
    gl_.bindBuffer(kGlArrayBuffer, 0);
}

const WebGl2Device::ProgramCache& WebGl2Device::ensureProgram(ShaderId id) {
    auto& pc = programs_[static_cast<size_t>(id)];
    if (pc.program != 0) return pc;
    const auto pair = gles::getShaderSources(id);
    pc.program = linkProgram(pair.vertexSrc, pair.fragmentSrc);
    if (pc.program == 0) return pc;
    pc.attr0 = gl_.getAttribLocation(pc.program, "aPosition");
    if (id == ShaderId::kSkinMask) {
        pc.attr1 = gl_.getAttribLocation(pc.program, "aMaskWeights");
    } else if (id == ShaderId::kRetouchSkin || id == ShaderId::kRetouchEyeTeeth || id == ShaderId::kParticleBurst) {
        pc.attr1 = gl_.getAttribLocation(pc.program, "aSrcCoord");
        pc.attr2 = gl_.getAttribLocation(pc.program, "aRetouchAttr");
    } else if (id == ShaderId::kMakeupLayer || id == ShaderId::kMakeupLip || id == ShaderId::kMakeupIris) {
        pc.attr1 = gl_.getAttribLocation(pc.program, "aScreenUv");
        pc.attr2 = gl_.getAttribLocation(pc.program, "aMakeupUv");
    } else if (id == ShaderId::kHeadDepthShell || id == ShaderId::kAr3dMesh) {
        pc.attr1 = gl_.getAttribLocation(pc.program, "aTextureCoord");
        pc.attr2 = gl_.getAttribLocation(pc.program, "aDepth");
    } else {
        pc.attr1 = gl_.getAttribLocation(pc.program, "aTextureCoord");
    }
    char name[16];
    for (int i = 0; i < 4; ++i) {
        std::snprintf(name, sizeof(name), "uTex%d", i);
        pc.texLoc[i] = gl_.getUniformLocation(pc.program, name);
    }
    for (int i = 0; i < 16; ++i) {
        std::snprintf(name, sizeof(name), "uU%d", i);
        pc.uniLoc[i] = gl_.getUniformLocation(pc.program, name);
    }
    return pc;
}

void WebGl2Device::beginPass(RenderTargetId rt, int viewportW, int viewportH, LoadOp loadOp, const Vec4& clearColor) {
    if (!isReady()) return;
    auto it = rts_.find(rt);
    gl_.bindFramebuffer(kGlFramebuffer, (it != rts_.end()) ? it->second.fbo : 0);
    gl_.viewport(0, 0, viewportW, viewportH);
    gl_.disable(kGlDepthTest);
    gl_.disable(kGlScissorTest);
    if (loadOp == LoadOp::kClear) {
        gl_.clearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        gl_.clear(kGlColorBufferBit);
    }
}

void WebGl2Device::applyStateAndUniforms(
    const ProgramCache& pc, const RenderState& state,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    if (state.blendEnable) {
        gl_.enable(kGlBlend);
        gl_.blendFuncSeparate(kGlSrcAlpha, kGlOneMinusSrcAlpha, kGlZero, kGlOne);
    } else {
        gl_.disable(kGlBlend);
    }
    if (state.cullBack) {
        gl_.enable(kGlCullFace);
        gl_.frontFace(state.frontFaceCw ? kGlCw : kGlCcw);
        gl_.cullFace(kGlBack);
    } else {
        gl_.disable(kGlCullFace);
        gl_.frontFace(kGlCcw);
    }
    const uint8_t rgbMask = state.colorMaskAlphaOnly ? 0 : 1;
    gl_.colorMask(rgbMask, rgbMask, rgbMask, 1);

    gl_.useProgram(pc.program);
    for (int i = 0; i < textures.count; ++i) {
        if (pc.texLoc[i] >= 0 && textures.slots[i] != kInvalidTexture) {
            gl_.activeTexture(kGlTexture0 + static_cast<uint32_t>(i));
            gl_.bindTexture(kGlTexture2D, unwrapWebGlTexture(textures.slots[i]));
            gl_.uniform1i(pc.texLoc[i], i);
        }
    }
    for (int i = 0; i < uniforms.count; ++i) {
        if (pc.uniLoc[i] >= 0) {
            const auto& v = uniforms.v[i];
            gl_.uniform4f(pc.uniLoc[i], v.x, v.y, v.z, v.w);
        }
    }
}

void WebGl2Device::restoreSharedState() {
    gl_.disable(kGlCullFace);
    gl_.frontFace(kGlCcw);
    gl_.disable(kGlBlend);
    gl_.colorMask(1, 1, 1, 1);
    gl_.bindBuffer(kGlArrayBuffer, 0);
    gl_.bindBuffer(kGlElementArrayBuffer, 0);
    for (int i = 3; i >= 0; --i) {
        gl_.activeTexture(kGlTexture0 + static_cast<uint32_t>(i));
        gl_.bindTexture(kGlTexture2D, 0);
    }
}

void WebGl2Device::drawFullScreenQuad(
    ShaderId shader, const RenderState& state,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    if (!isReady()) return;
    const auto& pc = ensureProgram(shader);
    if (!pc.program) return;
    ensureQuadVbo();
    applyStateAndUniforms(pc, state, textures, uniforms);

    gl_.bindBuffer(kGlArrayBuffer, quadVbo_);
    if (pc.attr0 >= 0) {
        gl_.enableVertexAttribArray(static_cast<uint32_t>(pc.attr0));
        gl_.vertexAttribPointer(static_cast<uint32_t>(pc.attr0), 2, kGlFloat, 0,
                                4 * static_cast<int>(sizeof(float)), reinterpret_cast<const void*>(0));
    }
    if (pc.attr1 >= 0) {
        gl_.enableVertexAttribArray(static_cast<uint32_t>(pc.attr1));
        gl_.vertexAttribPointer(static_cast<uint32_t>(pc.attr1), 2, kGlFloat, 0,
                                4 * static_cast<int>(sizeof(float)),
                                reinterpret_cast<const void*>(2 * sizeof(float)));
    }
    gl_.drawArrays(kGlTriangleStrip, 0, 4);
    if (pc.attr0 >= 0) gl_.disableVertexAttribArray(static_cast<uint32_t>(pc.attr0));
    if (pc.attr1 >= 0) gl_.disableVertexAttribArray(static_cast<uint32_t>(pc.attr1));
    restoreSharedState();
}

void WebGl2Device::drawIndexed(
    ShaderId shader, const RenderState& state,
    BufferId vertexBuffer, BufferId indexBuffer,
    int indexOffset, int indexCount, VertexLayout layout,
    const TextureBindings& textures, const UniformBlock& uniforms) {
    if (!isReady() || indexCount <= 0 || indexOffset < 0) return;
    const auto& pc = ensureProgram(shader);
    auto vbIt = buffers_.find(vertexBuffer);
    auto ibIt = buffers_.find(indexBuffer);
    if (!pc.program || vbIt == buffers_.end() || ibIt == buffers_.end()) return;

    applyStateAndUniforms(pc, state, textures, uniforms);
    gl_.bindBuffer(kGlArrayBuffer, vbIt->second.glBuf);
    gl_.bindBuffer(kGlElementArrayBuffer, ibIt->second.glBuf);

    auto bindAttr = [&](int loc, int comps, int strideFloats, int offsetFloats) {
        if (loc >= 0) {
            gl_.enableVertexAttribArray(static_cast<uint32_t>(loc));
            gl_.vertexAttribPointer(
                static_cast<uint32_t>(loc), comps, kGlFloat, 0,
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
        case VertexLayout::kAr3dPos2Uv2Z1:
            bindAttr(pc.attr0, 2, 5, 0);
            bindAttr(pc.attr1, 2, 5, 2);
            bindAttr(pc.attr2, 1, 5, 4);
            break;
    }

    gl_.drawElements(
        kGlTriangles, indexCount, kGlUnsignedShort,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(indexOffset) * sizeof(uint16_t)));

    if (pc.attr0 >= 0) gl_.disableVertexAttribArray(static_cast<uint32_t>(pc.attr0));
    if (pc.attr1 >= 0) gl_.disableVertexAttribArray(static_cast<uint32_t>(pc.attr1));
    if (pc.attr2 >= 0) gl_.disableVertexAttribArray(static_cast<uint32_t>(pc.attr2));
    restoreSharedState();
}

void WebGl2Device::endPass() {
    if (!isReady()) return;
    gl_.bindFramebuffer(kGlFramebuffer, 0);
}

}  // namespace livebeauty::rhi::webgl
