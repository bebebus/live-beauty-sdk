package io.bebebus.livebeauty

import android.opengl.GLES20.*

/**
 * 轻量无外部依赖的 OpenGL ES 2.0 着色器编译与程序链接工具。
 */
internal object GlUtil {
    fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = loadShader(GL_VERTEX_SHADER, vertexSource)
        if (vertexShader == 0) return 0
        val fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentSource)
        if (fragmentShader == 0) {
            glDeleteShader(vertexShader)
            return 0
        }
        val program = glCreateProgram()
        if (program == 0) {
            glDeleteShader(vertexShader)
            glDeleteShader(fragmentShader)
            return 0
        }
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragmentShader)
        glLinkProgram(program)
        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)
        val linkStatus = IntArray(1)
        glGetProgramiv(program, GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] != GL_TRUE) {
            val log = glGetProgramInfoLog(program)
            glDeleteProgram(program)
            throw IllegalStateException("GL program link failed: $log")
        }
        return program
    }

    private fun loadShader(shaderType: Int, source: String): Int {
        val shader = glCreateShader(shaderType)
        if (shader == 0) return 0
        glShaderSource(shader, source)
        glCompileShader(shader)
        val compiled = IntArray(1)
        glGetShaderiv(shader, GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            throw IllegalStateException("GL shader compile failed ($shaderType): $log")
        }
        return shader
    }
}
