import re
import subprocess
import sys
import os

def parse_instruction(line):
    parts = re.split(r':\t|\t', line.strip())
    if len(parts) >= 3:
        machine_code = parts[1]
        instruction = ' '.join(parts[2:])
    return machine_code, instruction


def parse_debug_line(file_path):
    try:
        objdump_cmd = f"objdump -d {file_path}"
        objdump_output = subprocess.check_output(objdump_cmd, shell=True).decode()

        smstart_id = 0
        machineCodes = []
        instructions = []
        i = 0 # bin smstart line
        for line in objdump_output.splitlines():
            match = re.match(r'\s*([0-9a-f]+):', line)
            if match:
                machineCode, instruction = parse_instruction(line)
                machineCodes.append(machineCode)
                instructions.append(instruction)
                if instruction == "smstart":
                    smstart_id = i
                i += 1

        return machineCodes, instructions, smstart_id
    
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return {}

def clean_line(line_content):
    clean_content = re.sub(r'//.*$', '', line_content.strip(), flags=re.MULTILINE)
    clean_content = re.sub(r'/\*.*?\*/', '', clean_content.strip(), flags=re.DOTALL)
    return clean_content

z_reg_pattern = r'z\d+\.'
za_reg_pattern = r'za\d+\.'
p_reg_pattern = r'p\d+'
pn_reg_pattern = r'pn\d+'
za_reg_pattern2 = r'za'


def need_convert(line):
    has_z = bool(re.search(z_reg_pattern, line))
    has_za = bool(re.search(za_reg_pattern, line))
    has_p = bool(re.search(p_reg_pattern, line))
    has_pn = bool(re.search(pn_reg_pattern, line))
    has_zero = "zero" in line # zero {za.s}
    has_cnth = "cnth" in line # cnth w14
    has_addvl = "addvl" in line # addvl x12, x12, #1
    is_smstart = "smstart" == line
    is_smstop = "smstop" == line
    has_zt0 = "zt0" in line # ldr zt0, [x0]

    to_convert = has_z or has_za or has_p or has_pn or has_zero or has_cnth or has_addvl or is_smstart or is_smstop or has_zt0
    return to_convert

def create_mapping(source_file, binary_file):
    with open(source_file, 'r') as f:
        source_lines = f.readlines()

    machineCodes, instructions, smstart_line_number = parse_debug_line(binary_file)
    total_source_lines = len(source_lines)
    bin_line_number = 0

    source_smstart_number = 0
    for line in source_lines:
        pure_content = clean_line(line)
        if pure_content.startswith('asm_function'):
            break
        source_smstart_number += 1

    src_line_number = source_smstart_number + 1 # start from the next line of 'asm_function'

    while src_line_number < total_source_lines:
        line_content = source_lines[src_line_number]
        pure_content = clean_line(line_content)


        while not need_convert(pure_content):
            # delete the comment lines
            while pure_content.startswith('/*'):
                src_line_number += 1
                if src_line_number >= total_source_lines:
                    break
                line_content = source_lines[src_line_number]
                pure_content = clean_line(line_content)
                while not pure_content.startswith('*/'):
                    src_line_number += 1
                    if src_line_number >= total_source_lines:
                        break
                    line_content = source_lines[src_line_number]
                    pure_content = clean_line(line_content)
                if src_line_number >= total_source_lines:
                    break
                src_line_number += 1
                if src_line_number >= total_source_lines:
                    break
                line_content = source_lines[src_line_number]
                pure_content = clean_line(line_content)

            if need_convert(pure_content):
                break
            src_line_number += 1
            if src_line_number >= total_source_lines:
                break
            line_content = source_lines[src_line_number]
            pure_content = clean_line(line_content)
        
        while (pure_content.startswith('.inst')):
            src_line_number += 1
            if src_line_number >= total_source_lines:
                break
            line_content = source_lines[src_line_number]
            pure_content = clean_line(line_content)
        if src_line_number >= total_source_lines:
            break
        op = pure_content.split(' ')[0] 

        while not need_convert(instructions[bin_line_number]):
            bin_line_number += 1

        while (op != instructions[bin_line_number].split(' ')[0]):
            case1 = (op == 'mova' and instructions[bin_line_number].split(' ')[0] == 'mov')
            case2 = (op == 'dup' and instructions[bin_line_number].split(' ')[0] == 'mov')
            case3 = (op == 'eor' and instructions[bin_line_number].split(' ')[0] == 'not')
            if case1 or case2 or case3:
                break
            bin_line_number += 1
        # print(pure_content, instructions[bin_line_number])
        id0 = line_content.find(pure_content)
        new_line = line_content[:id0] + ".inst 0x" + machineCodes[bin_line_number] + " // " + pure_content
        if line_content[id0 + len(pure_content):] != "\n":
            new_line = new_line + " " + line_content[id0 + len(pure_content):]
        else:
            new_line = new_line + "\n"
        source_lines[src_line_number] = new_line
        bin_line_number += 1
        src_line_number += 1

        
        if pure_content == "smstop":
            break

    return source_lines

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python obj2asm.py src.asm [dst.asm]')
    source_file = sys.argv[1]
    binary_file = 'temp.o'
    current_directory = os.path.dirname(os.path.abspath(__file__))
    level1_directory = os.path.dirname(current_directory)
    level2_directory = os.path.dirname(level1_directory)
    header_directory = os.path.join(level2_directory, 'source/backend/cpu/arm')
    build_cmd = f"gcc -c -fno-tree-vectorize -march=armv8.2-a+sve+sve2+sme+sme2+fp16 {source_file} -I{header_directory} -o {binary_file}"
    subprocess.check_output(build_cmd, shell=True).decode()

    if len(sys.argv) > 3:
        dst_file = sys.argv[3]
    else:
        dst_file = source_file

    new_lines = create_mapping(source_file, binary_file)

    file = open(dst_file, 'wt')
    file.writelines(new_lines)
    file.close()

    rm_cmd = f"rm {binary_file}"
    subprocess.check_output(rm_cmd, shell=True).decode()
    print(f"Mapping completed. Output written to {dst_file}")
    print("Please check the output file for the mapping results.")