import re
import sys
import subprocess
import os

import difflib

import re
import sys
import difflib

SIMILARITY_THRESHOLD = 0.85

SYNONYM_RULES = [
    # 规则 1 & 2: 简单的 dup/mov 同义词
    (re.compile(r'^(dup)\s+(.*)$'), "mov {1}", "dup -> mov"),
    (re.compile(r'^(mov)\s+(z\d+\.q,\s*z\d+\.q\[\d+\])$'), "dup {1}", "mov z.q -> dup z.q"),

    # 规则 3 & 4: 仅限 ptrue 的 #N <-> vlN
    (re.compile(r'^(ptrue\s+p\d+\.[bshd]\s*,)\s*#(\d+)(?:\.0)?$'), "{0} vl{1}", "ptrue..., #${N} -> ..."),
    (re.compile(r'^(ptrue\s+p\d+\.[bshd]\s*,)\s*vl(\d+)$'), "{0} #{1}", "ptrue..., vl${N} -> ..."),

    # 规则 5 & 6 (新增): 处理 z<R>.<S>[0] <-> <S><R> 的等价关系
    # 例如: z29.h[0] -> h29
    (
        re.compile(r'^(.*,)\s*z(\d+)\.([bshd])\[0\]$'),
        lambda m: f"{m.group(1)} {m.group(3)}{m.group(2)}",
        "..., z<R>.<S>[0] -> ..., <S><R>"
    ),
    # # 例如: h29 -> z29.h[0]
    (
        re.compile(r'^(.*,)\s*([bshd])(\d+)$'),
        lambda m: f"{m.group(1)} z{m.group(3)}.{m.group(2)}[0]",
        "..., <S><R> -> ..., z<R>.<S>[0]"
    ),
]

def get_canonical_form(line):
    line = line.split('//')[0].split('@')[0]
    line = re.sub(r'\s+', ' ', line).strip().lower()
    if not line: return ""
    def number_replacer(match):
        num_str = match.group(1)
        try:
            num_val = float(num_str)
            if '.' not in num_str and 'e' not in num_str.lower(): return f'#{int(num_val)}'
            return f'#{str(num_val)}'
        except ValueError: return match.group(0)
    line = re.sub(r'#([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', number_replacer, line)
    return line

def should_be_converted(instruction_line):
    """
    根据用户定义的规则，判断一条指令是否应该被转换为 .inst 格式。
    *** 此函数已更新以包含 'cnth' 和 'pn' 寄存器 ***
    """
    # 规则1：如果指令助记符是特殊指令之一，则必须转换
    parts = instruction_line.split()
    if parts and parts[0] in {'addvl', 'cnth', 'cntw', 'smstart', 'smstop'}:
        return True

    # 规则2：如果包含 p, pn, z, 或 za 寄存器，则必须转换
    # 正则表达式已更新以包含 pn<数字>
    if re.search(r'\b(p\d+|pn\d+|z\d+|za+|zt0)', instruction_line):
        return True

    # 如果以上条件都不满足，则不转换
    return False

def parse_objdump(objdump_file):
    pattern = re.compile(r'^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(.+)$')
    instruction_map = {}
    with open(objdump_file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                hex_code, asm_instruction = match.group(1), match.group(2).strip()
                canonical_key = get_canonical_form(asm_instruction)
                if canonical_key: instruction_map[canonical_key] = hex_code
    return instruction_map

def generate_equivalent_instructions(canonical_line):
    equivalents = {canonical_line} # 使用集合以自动去重
    # 迭代处理，因为一个规则的输出可能是另一个规则的输入
    items_to_process = [canonical_line]
    while items_to_process:
        line = items_to_process.pop(0)
        for pattern, replacement, _ in SYNONYM_RULES:
            match = pattern.match(line)
            if match:
                if callable(replacement):
                    # 如果替换规则是函数，则调用它
                    new_instr = replacement(match)
                else:
                    # 否则，使用字符串格式化
                    new_instr = replacement.format(*match.groups())

                if new_instr not in equivalents:
                    equivalents.add(new_instr)
                    items_to_process.append(new_instr)
    return list(equivalents)

def expand_register_range(instruction):
    """
    查找并展开指令中的寄存器范围，如 {z26.s-z27.s} -> {z26.s,z27.s}。
    {z0.s-z3.s} 会被展开为 {z0.s,z1.s,z2.s,z3.s}。
    """
    # 正则表达式模式:
    # \{ \s*             - 匹配 '{' 和可选空格
    # ([a-zA-Z])(\d+)    - 捕获组1(前缀), 捕获组2(起始编号)
    # (\.\w+)            - 捕获组3(后缀)
    # \s*-\s*            - 匹配 '-' 和可选空格
    # \1(\d+)\3          - 匹配相同的组1(前缀), 捕获组4(结束编号), 相同的组3(后缀)
    # \s* \}             - 匹配可选空格和 '}'
    pattern = re.compile(r'\{\s*([a-zA-Z])(\d+)(\.\w+)\s*-\s*\1(\d+)\3\s*\}')

    # 定义一个替换函数，用于生成展开后的列表
    def replacer(match):
        prefix, start_num_str, suffix, end_num_str = match.groups()
        start_num, end_num = int(start_num_str), int(end_num_str)

        # 确保范围是有效的
        if start_num >= end_num:
            return match.group(0) # 如果范围无效，则不替换

        # 使用列表推导生成所有寄存器名
        regs = [f"{prefix}{i}{suffix}" for i in range(start_num, end_num + 1)]

        # 将列表连接成一个无空格的字符串，并用花括号包裹
        # 输出如: {z26.s,z27.s}
        return f"{{{','.join(regs)}}}"

    # 使用 re.sub 和我们的替换函数来执行替换
    return pattern.sub(replacer, instruction)

def normalize_instruction(instruction):
    """
    对汇编指令进行语义规范化。
    - 步骤 0: 移除分号后的注释。
    - 步骤 1: 展开寄存器范围 (例如, z26.s-z27.s)。
    - 步骤 2: 规范化数字，统一转为十进制格式。
    - 步骤 3: 仅保留助记符后的第一个空格，移除所有其他空格。
    """
    # 步骤 0: 移除注释
    instruction = instruction.split(';')[0].strip()

    # 步骤 1: 展开寄存器范围
    instruction = expand_register_range(instruction)

    # 步骤 2: 规范化数字
    match = re.search(r'#\s*(0x[0-9a-fA-F]+|[0-9]+)', instruction)
    if match:
        number_str = match.group(1)
        try:
            decimal_value = int(number_str, 0)
            instruction = instruction.replace(match.group(0), f'#{decimal_value}')
        except ValueError:
            pass

    # 步骤 3: 规范化空格
    parts = instruction.split(' ', 1)
    if len(parts) == 2:
        mnemonic = parts[0]
        operands = parts[1]
        operands_no_space = operands.replace(' ', '')
        return f"{mnemonic} {operands_no_space}"
    else:
        return instruction

def find_best_match(source_line, instruction_map):
    matcher = difflib.SequenceMatcher(None, normalize_instruction(source_line))
    best_match_key, highest_score = None, 0.0
    for key in instruction_map.keys():
        keyNormalized = normalize_instruction(key)
        matcher.set_seq2(keyNormalized)
        score = matcher.ratio()
        if score > highest_score: highest_score, best_match_key = score, key
    return best_match_key, highest_score

def find_mnemonic_matches(source_line, instruction_map):
    """
    新增函数：查找所有指令助记符相同的指令。
    """
    source_mnemonic = source_line.split()[0] if source_line else ""
    if not source_mnemonic:
        return []

    matches = []
    for key in instruction_map.keys():
        if key.split()[0] == source_mnemonic:
            matches.append(key)
    return matches

def process_assembly_file(s_file_path, instruction_map, output_file_path):
    """
    主处理函数，已集成新的过滤逻辑和增强的错误报告。
    """
    with open(s_file_path, 'r') as f_in, open(output_file_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            match = re.match(r'^(\s*)(.*)$', line)
            indentation, content_with_comment = match.group(1), match.group(2).strip()

            if not content_with_comment or content_with_comment.startswith(('.', '//', '#', '@')) or content_with_comment.endswith(':'):
                f_out.write(line)
                continue

            canonical_content = get_canonical_form(content_with_comment)
            if not should_be_converted(canonical_content):
                f_out.write(line)
                continue

            found_match = False
            hex_code = None
            equivalent_candidates = generate_equivalent_instructions(canonical_content)
            for candidate in equivalent_candidates:
                if candidate in instruction_map:
                    hex_code = instruction_map[candidate]
                    found_match = True
                    if candidate != canonical_content:
                        print(f"提示 (行 {line_num}): 使用等价匹配 '{canonical_content}' -> '{candidate}'")
                    break

            if found_match:
                new_line = f"{indentation}.inst 0x{hex_code}  // {content_with_comment}\n"
                f_out.write(new_line)
                continue

            # --- 全新的、增强的错误报告逻辑 ---
            print("--------------------------------------------------")

            # 报告1: 全局最相似的匹配
            best_match_key, score = find_best_match(canonical_content, instruction_map)
            # if best_match_key:
            #     print(f"  -> 全局最相似的匹配是 '{best_match_key}' (相似度: {score:.2%})")

            # 报告2: 所有助记符相同的匹配
            # mnemonic_matches = find_mnemonic_matches(canonical_content, instruction_map)
            # if mnemonic_matches:
            #     source_mnemonic = canonical_content.split()[0]
            #     print(f"  -> 在 Objdump 中找到以下助记符为 '{source_mnemonic}' 的指令:")
            #     for m_match in mnemonic_matches:
            #         print(f"     - '{m_match}'")

            if score > SIMILARITY_THRESHOLD:
                print(f"警告 (行 {line_num}): '{content_with_comment}' 与 '{best_match_key}' 的相似度为 {score:.2%}，这里同样进行替换。请检查是否正确。")
                new_line = f"{indentation}.inst 0x{instruction_map[best_match_key]}  // {content_with_comment}\n"
                f_out.write(new_line)
            else:
                print(f"错误 (行 {line_num}): 无法为 '{content_with_comment}' 找到任何直接或等价的匹配项，最相似匹配是 {best_match_key}, 相似分是{score} 请检查指令或手动添加支持。")
                f_out.write(line) # 保持原样

def main():
    if len(sys.argv) != 2:
        print("用法: python arm2binary.py <原始S文件>")
        print("例如: python arm2binary.py MNNPackedMatMulRemainFP32_SME2.S")
        sys.exit(1)

    s_file = sys.argv[1]
    output_file = s_file.replace('.S', '_with_inst.S')

    # 生成临时objdump文件名
    binary_file = 'temp.o'
    objdump_file = 'temp_objdump.txt'
    current_directory = os.path.dirname(os.path.abspath(__file__))
    level1_directory = os.path.dirname(current_directory)
    level2_directory = os.path.dirname(level1_directory)
    header_directory = os.path.join(level2_directory, 'source/backend/cpu/arm')
    build_cmd = f"gcc -c -fno-tree-vectorize -march=armv8.2-a+sve+sve2+sme+sme2+fp16 {s_file} -I{header_directory} -o {binary_file}"
    subprocess.check_output(build_cmd, shell=True).decode()
    objdump_cmd = f"objdump -d {binary_file} > {objdump_file}"
    subprocess.check_output(objdump_cmd, shell=True).decode()

    print(f"1. 正在解析机器码...")
    instruction_map = parse_objdump(objdump_file)
    if not instruction_map:
        print("错误：未能从objdump文件中解析出任何指令。请检查文件内容。")
        sys.exit(1)
    print(f"   ...成功解析 {len(instruction_map)} 条指令。")

    print(f"2. 正在处理汇编文件...")
    process_assembly_file(s_file, instruction_map, output_file)
    copy_cmd = f"cp {output_file} {s_file}"
    subprocess.check_output(copy_cmd, shell=True).decode()
    print("   ...处理完成！")

    rm_cmd = f"rm {binary_file} {objdump_file} {output_file}"
    subprocess.check_output(rm_cmd, shell=True).decode()


if __name__ == "__main__":
    main()