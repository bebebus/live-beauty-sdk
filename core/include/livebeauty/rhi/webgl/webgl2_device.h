#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "livebeauty/rhi/rhi_device.h"

namespace livebeauty::rhi::webgl {

/**
 * Portable WebGL2 / OpenGL ES 3.0 function-pointer dispatch table.
 * - On Emscripten (`__EMSCRIPTEN__`), `WebGl2DispatchTable::emscriptenDefault()` binds directly to
 *   `<GLES3/gl3.h>` and `<emscripten/html5_webgl.h>`.
 * - On desktop CI/hosts, tests can inject a mock/headless WebGL2 dispatch table to verify WebGL2
 *   RHI resource lifecycle, `GL_R8` masks, `EXT_color_buffer_float` `GL_RGBA16F` negotiation, and
 *   all 31 shader programs without requiring a browser runtime.
 */
struct WebGl2DispatchTable {
    void (*genTextures)(int n, uint32_t* ids) = nullptr;
    void (*deleteTextures)(int n, const uint32_t* ids) = nullptr;
    void (*bindTexture)(uint32_t target, uint32_t tex) = nullptr;
    void (*activeTexture)(uint32_t unit) = nullptr;
    void (*texParameteri)(uint32_t target, uint32_t pname, int param) = nullptr;
    void (*pixelStorei)(uint32_t pname, int param) = nullptr;
    void (*texImage2D)(uint32_t target, int level, int internalFmt, int w, int h, int border,
                       uint32_t fmt, uint32_t type, const void* pixels) = nullptr;
    void (*texSubImage2D)(uint32_t target, int level, int xoff, int yoff, int w, int h,
                          uint32_t fmt, uint32_t type, const void* pixels) = nullptr;

    void (*genFramebuffers)(int n, uint32_t* ids) = nullptr;
    void (*deleteFramebuffers)(int n, const uint32_t* ids) = nullptr;
    void (*bindFramebuffer)(uint32_t target, uint32_t fbo) = nullptr;
    void (*framebufferTexture2D)(uint32_t target, uint32_t attachment, uint32_t texTarget,
                                 uint32_t tex, int level) = nullptr;
    uint32_t (*checkFramebufferStatus)(uint32_t target) = nullptr;

    void (*genBuffers)(int n, uint32_t* ids) = nullptr;
    void (*deleteBuffers)(int n, const uint32_t* ids) = nullptr;
    void (*bindBuffer)(uint32_t target, uint32_t buf) = nullptr;
    void (*bufferData)(uint32_t target, ptrdiff_t size, const void* data, uint32_t usage) = nullptr;

    uint32_t (*createShader)(uint32_t type) = nullptr;
    void (*deleteShader)(uint32_t shader) = nullptr;
    void (*shaderSource)(uint32_t shader, int count, const char* const* str, const int* len) = nullptr;
    void (*compileShader)(uint32_t shader) = nullptr;
    void (*getShaderiv)(uint32_t shader, uint32_t pname, int* out) = nullptr;

    uint32_t (*createProgram)() = nullptr;
    void (*deleteProgram)(uint32_t prog) = nullptr;
    void (*attachShader)(uint32_t prog, uint32_t shader) = nullptr;
    void (*linkProgram)(uint32_t prog) = nullptr;
    void (*getProgramiv)(uint32_t prog, uint32_t pname, int* out) = nullptr;
    void (*useProgram)(uint32_t prog) = nullptr;

    int (*getAttribLocation)(uint32_t prog, const char* name) = nullptr;
    int (*getUniformLocation)(uint32_t prog, const char* name) = nullptr;
    void (*enableVertexAttribArray)(uint32_t idx) = nullptr;
    void (*disableVertexAttribArray)(uint32_t idx) = nullptr;
    void (*vertexAttribPointer)(uint32_t idx, int size, uint32_t type, uint8_t normalized,
                                int stride, const void* offset) = nullptr;
    void (*uniform1i)(int loc, int v0) = nullptr;
    void (*uniform4f)(int loc, float x, float y, float z, float w) = nullptr;

    void (*viewport)(int x, int y, int w, int h) = nullptr;
    void (*enable)(uint32_t cap) = nullptr;
    void (*disable)(uint32_t cap) = nullptr;
    void (*blendFuncSeparate)(uint32_t srcRgb, uint32_t dstRgb, uint32_t srcA, uint32_t dstA) = nullptr;
    void (*frontFace)(uint32_t mode) = nullptr;
    void (*cullFace)(uint32_t mode) = nullptr;
    void (*colorMask)(uint8_t r, uint8_t g, uint8_t b, uint8_t a) = nullptr;
    void (*clearColor)(float r, float g, float b, float a) = nullptr;
    void (*clear)(uint32_t mask) = nullptr;
    void (*drawArrays)(uint32_t mode, int first, int count) = nullptr;
    void (*drawElements)(uint32_t mode, int count, uint32_t type, const void* indices) = nullptr;
    uint32_t (*getError)() = nullptr;
    bool (*enableExtension)(const char* extName) = nullptr;

#ifdef __EMSCRIPTEN__
    static WebGl2DispatchTable emscriptenDefault();
#endif
};

/**
 * WebGL2 / Emscripten WebAssembly RHI backend (`IRhiDevice`).
 * Enables running the identical C++17 `LiveBeautyEngine` + 31-shader `RenderPipeline` inside web browsers
 * (`HTMLVideoElement` / `VideoFrame` / `OffscreenCanvas` WebRTC & WebCodecs pipelines).
 */
class WebGl2Device final : public IRhiDevice {
public:
#ifdef __EMSCRIPTEN__
    WebGl2Device();
#endif
    explicit WebGl2Device(const WebGl2DispatchTable& gl);
    ~WebGl2Device() override;

    void releaseAll(bool contextAlive);
    bool isReady() const;
    bool supportsHalfFloatColorBuffer() const { return colorBufferFloatExt_; }

    /** Wrap an external WebGLTexture ID (e.g. uploaded from HTMLVideoElement / VideoFrame). */
    TextureId wrapExternalTexture(uint32_t webglTexId);
    uint32_t unwrapWebGlTexture(TextureId tex) const;

    RenderTargetId createRenderTarget(int width, int height, TextureFormat format) override;
    void destroyRenderTarget(RenderTargetId rt) override;
    TextureId textureOf(RenderTargetId rt) const override;

    TextureId createTexture2D(int width, int height, TextureFormat format, const uint8_t* pixels) override;
    void updateTexture2D(TextureId tex, int width, int height, TextureFormat format, const uint8_t* pixels) override;
    void destroyTexture(TextureId tex) override;

    BufferId createDynamicVertexBuffer(size_t byteSize) override;
    void updateVertexBuffer(BufferId buf, const void* data, size_t byteSize) override;
    BufferId createStaticIndexBuffer(const uint16_t* indices, size_t count) override;
    void destroyBuffer(BufferId buf) override;

    using IRhiDevice::beginPass;
    void beginPass(RenderTargetId rt, int viewportW, int viewportH, LoadOp loadOp, const Vec4& clearColor = {}) override;
    void drawFullScreenQuad(
        ShaderId shader, const RenderState& state,
        const TextureBindings& textures, const UniformBlock& uniforms) override;
    void drawIndexed(
        ShaderId shader, const RenderState& state,
        BufferId vertexBuffer, BufferId indexBuffer,
        int indexOffset, int indexCount, VertexLayout layout,
        const TextureBindings& textures, const UniformBlock& uniforms) override;
    void endPass() override;

private:
    struct ProgramCache {
        uint32_t program = 0;
        int attr0 = -1;
        int attr1 = -1;
        int attr2 = -1;
        std::array<int, 4> texLoc = {-1, -1, -1, -1};
        std::array<int, 16> uniLoc{};
    };
    struct RtRecord {
        uint32_t fbo = 0;
        TextureId texId = kInvalidTexture;
        int width = 0;
        int height = 0;
    };
    struct TexRecord {
        uint32_t glTex = 0;
        bool external = false;
    };
    struct BufRecord {
        uint32_t glBuf = 0;
    };

    uint32_t compileShader(uint32_t type, const char* src);
    uint32_t linkProgram(const char* vSrc, const char* fSrc);
    const ProgramCache& ensureProgram(ShaderId id);
    void ensureQuadVbo();
    void applyStateAndUniforms(const ProgramCache& pc, const RenderState& state,
                               const TextureBindings& textures, const UniformBlock& uniforms);
    void restoreSharedState();

    WebGl2DispatchTable gl_{};
    bool colorBufferFloatExt_ = false;
    uint32_t nextId_ = 1;
    uint32_t quadVbo_ = 0;
    std::array<ProgramCache, static_cast<size_t>(ShaderId::kCount)> programs_{};
    std::unordered_map<RenderTargetId, RtRecord> rts_;
    std::unordered_map<TextureId, TexRecord> textures_;
    std::unordered_map<BufferId, BufRecord> buffers_;
};

}  // namespace livebeauty::rhi::webgl
