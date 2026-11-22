package io.bebebus.livebeauty

import java.io.File

/**
 * 读取共享向量 reference/beauty_vectors.json（规格 v1 两端一致性）。JVM 单测里 android.jar 的 org.json 只是桩，
 * 又不新增依赖，所以这里带一个只够读该文件的最小 JSON 解析器：数字一律读成 Double。
 * Gradle 单测的工作目录是 android/ 模块目录。
 */
@Suppress("UNCHECKED_CAST")
internal object BeautyVectors {
    private val file = listOf(File("../../reference/beauty_vectors.json"), File("../reference/beauty_vectors.json"), File("reference/beauty_vectors.json")).firstOrNull { it.isFile } ?: File("../../reference/beauty_vectors.json")
    val root: Map<String, Any?> by lazy {
        check(file.isFile) { "找不到共享向量 ${file.absolutePath}" }
        Parser(file.readText(Charsets.UTF_8)).document() as Map<String, Any?>
    }
    fun section(name: String) = root[name] as Map<String, Any?>
    fun Any?.obj() = this as Map<String, Any?>
    fun Any?.list() = this as List<Any?>
    /** 向量里非有限值写成字符串 "NaN"/"Infinity"。 */
    fun Any?.number(): Double = when (this) {
        is Double -> this
        "NaN" -> Double.NaN
        "Infinity" -> Double.POSITIVE_INFINITY
        "-Infinity" -> Double.NEGATIVE_INFINITY
        else -> error("不是数字：$this")
    }
    fun Any?.floats() = list().map { it.number().toFloat() }.toFloatArray()

    private class Parser(private val text: String) {
        private var i = 0
        fun document(): Any? {
            val value = value(); skip()
            check(i == text.length) { "JSON 尾部有多余内容：$i" }
            return value
        }
        private fun skip() { while (i < text.length && text[i].isWhitespace()) i++ }
        private fun expect(c: Char) { skip(); check(text[i] == c) { "位置 $i 应为 $c" }; i++ }
        private fun value(): Any? {
            skip()
            return when (text[i]) {
                '{' -> obj()
                '[' -> array()
                '"' -> string()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                else -> number()
            }
        }
        private fun literal(word: String, value: Any?): Any? {
            check(text.startsWith(word, i)) { "位置 $i 不是 $word" }
            i += word.length
            return value
        }
        private fun obj(): Map<String, Any?> {
            expect('{')
            val out = LinkedHashMap<String, Any?>()
            skip()
            if (text[i] == '}') { i++; return out }
            while (true) {
                skip(); val key = string()
                expect(':'); out[key] = value()
                skip()
                if (text[i++] == '}') return out
                check(text[i - 1] == ',') { "位置 ${i - 1} 应为 , 或 }" }
            }
        }
        private fun array(): List<Any?> {
            expect('[')
            val out = ArrayList<Any?>()
            skip()
            if (text[i] == ']') { i++; return out }
            while (true) {
                out.add(value())
                skip()
                if (text[i++] == ']') return out
                check(text[i - 1] == ',') { "位置 ${i - 1} 应为 , 或 ]" }
            }
        }
        private fun string(): String {
            expect('"')
            val out = StringBuilder()
            while (true) {
                val c = text[i++]
                when (c) {
                    '"' -> return out.toString()
                    '\\' -> when (val e = text[i++]) {
                        'n' -> out.append('\n'); 't' -> out.append('\t'); 'r' -> out.append('\r')
                        'b' -> out.append('\b'); 'f' -> out.append('\u000c')
                        'u' -> { out.append(text.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                        else -> out.append(e)
                    }
                    else -> out.append(c)
                }
            }
        }
        private fun number(): Double {
            val start = i
            while (i < text.length && text[i] in "+-0123456789.eE") i++
            return text.substring(start, i).toDouble()
        }
    }
}
