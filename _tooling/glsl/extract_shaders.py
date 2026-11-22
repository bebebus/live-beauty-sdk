#!/usr/bin/env python3
"""从 VideoEffectsShaders.kt 提取 GLSL ES 1.00 原文（规格 v1 第 4.2 节「Android 着色器规则」）。

只认 `const val NAME = \"\"\"...\"\"\"`，把 `$NAME` / `${NAME}` 递归展开成同文件的其他常量，
输出与 Android 运行期看到的字符串逐字相同（不 dedent、不改写）。同时静态检查规格里编译器
查不出来的规则：模板只能引用同文件常量、文本里没有其他 `$`、不写 #version、不用保留标识符、
只写 ASCII、整个文件不调用 trimIndent()。

用法：python3 -I extract_shaders.py <VideoEffectsShaders.kt> [输出 JSON]
"""
import json
import re
import sys

BLOCK = re.compile(r'const\s+val\s+([A-Za-z_]\w*)\s*(?::\s*String\s*)?=\s*"""(.*?)"""', re.S)
TEMPLATE = re.compile(r'\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))')
# GLSL ES 1.00 保留或在部分驱动/后续版本里是关键字的名字，规格禁止作标识符。
RESERVED = ('packed', 'input', 'output', 'half', 'fixed', 'filter', 'sample')


class ShaderSourceError(Exception):
    pass


def strip_comments(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)


def extract(path):
    """返回 {常量名: 展开后的原文}，按文件中的出现顺序。违反规则时抛 ShaderSourceError。"""
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    raw = {}
    for match in BLOCK.finditer(source):
        if match.group(1) in raw:
            raise ShaderSourceError('常量 %s 重复定义' % match.group(1))
        raw[match.group(1)] = match.group(2)
    if not raw:
        raise ShaderSourceError('%s 里没有 const val 原始字符串' % path)
    if re.search(r'\.trimIndent\s*\(', source):
        raise ShaderSourceError('着色器文件调用了 trimIndent()，规格要求原文顶格')

    problems = []
    for name, text in raw.items():
        if re.search(r'[^\x00-\x7f]', text):
            problems.append('%s：着色器文本含非 ASCII 字符' % name)
        rest = TEMPLATE.sub('', text)
        if '$' in rest:
            problems.append('%s：文本里有不是 $NAME/${NAME} 模板的 $' % name)
        for ref in TEMPLATE.finditer(text):
            key = ref.group(1) or ref.group(2)
            if key not in raw:
                problems.append('%s：模板 $%s 不是本文件的 const 常量' % (name, key))
    if problems:
        raise ShaderSourceError('\n'.join(problems))

    expanded = {}

    def resolve(name, stack):
        if name in stack:
            raise ShaderSourceError('模板循环引用：%s' % ' -> '.join(stack + (name,)))
        if name not in expanded:
            expanded[name] = TEMPLATE.sub(
                lambda m: resolve(m.group(1) or m.group(2), stack + (name,)), raw[name])
        return expanded[name]

    result = {name: resolve(name, ()) for name in raw}
    for name, text in result.items():
        code = strip_comments(text)
        if re.search(r'^\s*#\s*version\b', code, re.M):
            problems.append('%s：不得写 #version' % name)
        for word in RESERVED:
            if re.search(r'\b%s\b' % word, code):
                problems.append('%s：%s 不得作标识符' % (name, word))
    if problems:
        raise ShaderSourceError('\n'.join(problems))
    return result


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 64
    try:
        shaders = extract(argv[1])
    except ShaderSourceError as error:
        print('着色器原文不合规：\n%s' % error, file=sys.stderr)
        return 1
    text = json.dumps(shaders, indent=1, ensure_ascii=False)
    if len(argv) == 3:
        with open(argv[2], 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
