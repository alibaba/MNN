import sys
import os
import re
import itertools

def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    return stdout.decode('utf-8')

def extract_macros(file_content):
    """提取宏定义"""
    macros = {}
    macros_num = {}
    ifdef_pattern = re.compile(r'#(ifdef)\s+(\w+)')
    ifndef_pattern = re.compile(r'#(ifndef)\s+(\w+)')
    if_pattern = re.compile(r'#(if)\s+(\w+)')
    elif_pattern = re.compile(r'#(elif)\s+(\w+)')
    defined_pattern = re.compile(r'(defined)\s+(\w+)')
    define_pattern = re.compile(r'#(define)\s+(\w+)')
    for match in ifdef_pattern.finditer(file_content):
        macro_type, macro_name = match.groups()
        if "LOCAL_SIZE" in macro_name:
            macros_num[macro_name] = {1, 2, 3, 4, 16}
        else:
            macros[macro_name] = None

    for match in ifndef_pattern.finditer(file_content):
        macro_type, macro_name = match.groups()
        if "LOCAL_SIZE" in macro_name:
             macros_num[macro_name] = {1, 2, 3, 4, 16}
        else:
            macros[macro_name] = None

    for match in if_pattern.finditer(file_content):
        macro_type, macro_name = match.groups()
        if macro_name != "defined":
            macros_num[macro_name] = {1, 2, 3, 4, 8}

    for match in elif_pattern.finditer(file_content):
        macro_type, macro_name = match.groups()
        if macro_name != "defined":
            macros_num[macro_name] = {1, 2, 3, 4, 8}

    for match in defined_pattern.finditer(file_content):
        macro_type, macro_name = match.groups()
        macros[macro_name] = None

    for match in define_pattern.finditer(file_content):
        macro_type, macro_name = match.groups()
        if macro_name in macros:
            del macros[macro_name]
        if macro_name in macros_num:
            del macros_num[macro_name]

    if "MNN_SUPPORT_FP16" in macros:
        del macros["MNN_SUPPORT_FP16"]
    
    #for macro_name, macro_value in macros.items():
        # Replace macro value
        #print(f"macro_name {macro_name}   macro_value {macro_value}")
    return [macros_num, macros]

def compile_with_macros(macros_all, operator_macro, extra_macro, filename, test_for_android):
    """
    Tries to compile the kernel given various macro values
    """
    macros_num = macros_all[0]
    macros = macros_all[1]
    float_option = "-DFLOAT=float -DFLOAT2=float2 -DFLOAT3=float3 -DFLOAT4=float4 -DFLOAT8=float8 -DFLOAT16=float16 -DCOMPUTE_FLOAT=float  -DCOMPUTE_FLOAT2=float2 -DCOMPUTE_FLOAT3=float3 -DCOMPUTE_FLOAT4=float4 -DCOMPUTE_FLOAT8=float8 -DCOMPUTE_FLOAT16=float16 -DCONVERT_COMPUTE_FLOAT=convert_float  -DCONVERT_COMPUTE_FLOAT2=convert_float2 -DCONVERT_COMPUTE_FLOAT3=convert_float3 -DCONVERT_COMPUTE_FLOAT4=convert_float4 -DCONVERT_COMPUTE_FLOAT8=convert_float8 -DCONVERT_COMPUTE_FLOAT16=convert_float16 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT=convert_float  -DCONVERT_FLOAT2=convert_float2 -DCONVERT_FLOAT3=convert_float3 -DCONVERT_FLOAT4=convert_float4 -DCONVERT_FLOAT8=convert_float8 -DCONVERT_FLOAT16=convert_float16"
    float_option += " -DINPUT_TYPE_I=float -DINPUT_TYPE_I4=float4 -DINPUT_TYPE=float -DINPUT_TYPE4=float4 -DINPUT_TYPE16=float16 -DRI_DATA=read_imagef -DOUTPUT_TYPE_I=float -DOUTPUT_TYPE_I4=float4 -DCONVERT_OUTPUT_I4=convert_float4 -DOUTPUT_TYPE=float -DOUTPUT_TYPE4=float4 -DOUTPUT_TYPE16=float16 -DCONVERT_OUTPUT4=convert_float4 -DCONVERT_OUTPUT16=convert_float16 -DWI_DATA=write_imagef"
    if filename in extra_macro:
        float_option += extra_macro[filename]
    keys = list(macros.keys())

    # 使用 itertools.product 生成所有可能的 0 和 1 的组合
    combinations = list(itertools.product([0, 1], repeat=len(keys)))

    options_normal = []
    # 获取普通的宏定义
    for combination in combinations:
        option_normal = float_option
        macros_out = dict(zip(keys, combination))
        for macro_name, macro_value in macros_out.items():
            if macro_value == 1:
                option_normal += f" -D{macro_name}={macro_value} "
        options_normal.append(option_normal)

    options_num_normal = []
    # 获取有多种取值的宏
    if len(macros_num) > 0 :
        option_num = ""
        for i in {1, 2, 3, 4, 8} :
            for macro_name in macros_num:
                option_num = f" -D{macro_name}={i} "
            for option_normal in options_normal:
                options_num_normal.append(option_normal + option_num)
    else:
        options_num_normal = options_normal

    options = []       
    # 获取OPERATOR的宏, 只需要验证第一个OPERATOR宏与其他宏的各种组合，其他的可以只验证一种组合
    if len(operator_macro) > 0 :
        has_combine = False
        for op in operator_macro:
            option_operator = f" -DOPERATOR={op} "
            if has_combine is True:
                options.append(options_num_normal[0] + option_operator)
            else:
                for option_num_normal in options_num_normal:
                    options.append(option_num_normal + option_operator)
                has_combine = True
    else:
        options = options_num_normal
                  

    with open('option.txt', 'w') as outfile:
        for option in options:
            outfile.write(option + '\n')

    if test_for_android == 1:
        run_cmd(['adb', 'push', 'kernel.cl', '/data/local/tmp/MNN'])
        run_cmd(['adb', 'push', 'option.txt', '/data/local/tmp/MNN'])
        run_cmd(['adb', 'push', 'OpenCLProgramBuildTest.out', '/data/local/tmp/MNN'])
        res = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./OpenCLProgramBuildTest.out %s'%(filename)])
        print(res)
    else:
        if sys.platform.startswith('win'):
            res = run_cmd(['OpenCLProgramBuildTest.exe', f'{filename}'])
            print(res)
        else:
            res = run_cmd(['./OpenCLProgramBuildTest.out', f'{filename}'])
            print(res)

def main():
    print("opencl_kernel_check.py path without_subgroup test_for_android")
    path = '.'
    without_subgroup = 1
    test_for_android = 0
    if len(sys.argv) > 1:
        path = sys.argv[1]

    if len(sys.argv) > 2:
        without_subgroup = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        test_for_android = int(sys.argv[3])
    
    binaryvec_operator = {"in0+in1", "in0*in1", "in0-in1", "in0>in1?in0:in1", "sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001))",
    "in0>in1?in0:in1", "convert_float4(-isgreater(in0,in1))", "convert_float4(-isless(in0,in1))", "convert_float4(-islessequal(in0,in1))", "convert_float4(-isgreaterequal(in0,in1))", "convert_float4(-isequal(in0,in1))",
    "floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))", "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1",
    "pow(in0,in1)", "(in0-in1)*(in0-in1)", "(in1==(float)0?(sign(in0)*(float4)(PI/2)):(atan(in0/in1)+(in1>(float4)0?(float4)0:sign(in0)*(float)PI)))", "convert_float4(-isnotequal(in0,in1))",
    "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1"}

    binary_operator = {"in0*in1", "in0+in1", "in0-in1", "sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001))", "in0>in1?in1:in0", "in0>in1?in0:in1", "(float)(isgreater(in0,in1))",
    "(float)(isless(in0,in1))", "(float)(islessequal(in0,in1))", "(float)(isgreaterequal(in0,in1))", "(float)(isequal(in0,in1))", "floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))",
    "in0-floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))*in1", "pow(in0,in1)", "(in0-in1)*(in0-in1)", 
    "(in1==(float)0?(sign(in0)*(float)(PI/2)):(atan(in0/in1)+(in1>(float)0?(float)0:sign(in0)*(float)PI)))", "(float)(isnotequal(in0,in1))", "in0-floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))*in1"}

    unary_operator = {"fabs(convert_float4(in))", "in*in", "rsqrt(convert_float4(in)>(float4)(0.000001)?convert_float4(in):(float4)(0.000001))", "-(in)", "exp(convert_float4(in))", "cos(convert_float4(in))", "sin(convert_float4(in))",
    "tan(convert_float4(in))", "atan(convert_float4(in))", "sqrt(convert_float4(in))", "ceil(convert_float4(in))", "native_recip(convert_float4(in))", "log1p(convert_float4(in))", "native_log(convert_float4(in)>(float4)(0.0000001)?convert_float4(in):(float4)(0.0000001))",
    "floor(convert_float4(in))", "in>(float4)((float)0)?(in+native_log(exp(convert_float4(-(in)))+(float4)(1.0))):(native_log(exp(convert_float4(in))+(float4)(1.0)))", "acosh(convert_float4(in))", "sinh(convert_float4(in))", "asinh(convert_float4(in))",
    "atanh(convert_float4(in))", "sign(convert_float4(in))", "round(convert_float4(in))", "cosh(convert_float4(in))", "erf(convert_float4(in))", "erfc(convert_float4(in))", "expm1(convert_float4(in))", "native_recip((float4)1+native_exp(convert_float4(-in)))",
    "(convert_float4(in)*native_recip((float4)1+native_exp(convert_float4(-in))))", "tanh(convert_float4(in))", "convert_float4(in)>(float4)(-3.0f)?(convert_float4(in)<(float4)(3.0f)?((convert_float4(in)*(convert_float4(in)+(float4)3.0f))/(float4)6.0f):convert_float4(in)):(float4)(0.0f)",
    "gelu(convert_float4(in))", "(erf(convert_float4(in)*(float4)0.7071067932881648)+(float4)1.0)*convert_float4(in)*(float4)0.5", "native_recip((float4)(1.0)+native_exp(convert_float4(-(in))))",
    "tanh(convert_float4(in))"}

    extra_macro = {}
    extra_macro["binary_subgroup_buf.cl"] = " -DINTEL_DATA=uint -DAS_INPUT_DATA=as_float -DAS_INPUT_DATA4=as_float4 -DAS_OUTPUT_DATA4=as_uint4 -DINTEL_SUB_GROUP_READ=intel_sub_group_block_read -DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4 -DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4"
    extra_macro["conv_2d_c1_subgroup_buf.cl"] = " -DINPUT_LINE_SIZE=16 -DINPUT_BLOCK_SIZE=16 -DINPUT_CHANNEL=16 -DFILTER_HEIGHT=3 -DFILTER_WIDTH=3 -DDILATION_HEIGHT=1 -DDILATION_WIDTH=1 -DSTRIDE_HEIGHT=1 -DSTRIDE_WIDTH=1"
    extra_macro["conv_2d_c16_subgroup_buf.cl"] = " -DINPUT_LINE_SIZE=16 -DINPUT_BLOCK_SIZE=16 -DINPUT_CHANNEL=16 -DFILTER_HEIGHT=3 -DFILTER_WIDTH=3 -DDILATION_HEIGHT=1 -DDILATION_WIDTH=1 -DSTRIDE_HEIGHT=1 -DSTRIDE_WIDTH=1"
    extra_macro["depthwise_conv2d_subgroup_buf.cl"] = " -DFILTER_HEIGHT=3 -DFILTER_WIDTH=3 -DDILATION_HEIGHT=1 -DDILATION_WIDTH=1 -DSTRIDE_HEIGHT=1 -DSTRIDE_WIDTH=1"
    extra_macro["matmul_local_buf.cl"] = " -DOPWM=64 -DOPWN=128 -DCPWK=8 -DOPTM=4 -DOPTN=8"
    extra_macro["pooling_subgroup_buf.cl"] = " -DINPUT_LINE_SIZE=16 -DSTRIDE_Y=2 -DSTRIDE_X=2 -DKERNEL_Y=4 -DKERNEL_X=4"
    extra_macro["reduction_buf.cl"] = " -DOPERATE(a,b)=(a+b) -DVALUE=0"
    extra_macro["reduction.cl"] = " -DOPERATE(a,b)=(a+b) -DVALUE=0"
    extra_macro["unary_subgroup_buf.cl"] = " -DINTEL_DATA=uint -DAS_INPUT_DATA=as_float -DAS_INPUT_DATA4=as_float4 -DAS_OUTPUT_DATA4=as_uint4 -DINTEL_SUB_GROUP_READ=intel_sub_group_block_read -DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4 -DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4"
    
    # 遍历当前目录的所有.cl文件
    for filename in os.listdir(path):
        if filename.endswith('.cl'):
            source_file = os.path.join(path, filename)
            with open(source_file, 'r') as file:
                file_content = file.read()
            
            with open('kernel.cl', 'w') as outfile:
                outfile.write(file_content)

            # 提取宏定义
            macros_all = extract_macros(file_content)
            # Compile with different macro values
            operator_macro = {}
            if filename == "binary_buf.cl" or filename == "binary.cl" or filename == "loop.cl" or filename == "binary_subgroup_buf.cl":
                operator_macro = binaryvec_operator
            elif filename == "loop_buf.cl":
                operator_macro = binary_operator
            elif filename == "unary_buf.cl" or filename == "unary.cl" or filename == "unary_subgroup_buf.cl":
                operator_macro = unary_operator
            
            if "subgroup" in filename and without_subgroup == 1:
                continue
            compile_with_macros(macros_all, operator_macro, extra_macro, filename, test_for_android)

if __name__ == "__main__":
    main()
