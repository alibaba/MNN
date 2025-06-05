import flatbuffers
from CLCache.Cache import Cache
from CLCache.BackendInfo import BackendInfo
        
def generate_cpp_header(buffer):
    cache = Cache.GetRootAs(buffer, 0)
    backends_len = cache.BackendsLength()

    opencl_tune_map = {}

    # 读取 BackendInfo 信息
    for i in range(backends_len):
        backend_info = cache.Backends(i)
        device_name = backend_info.DeviceName().decode("utf-8")
        
        autotuning_map = {}
        tunings_len = backend_info.TuningsLength()

        for j in range(tunings_len):
            tuning = backend_info.Tunings(j)
            key = tuning.Key().decode("utf-8")
            global_size = list(tuning.GloablSize(j) for j in range(tuning.GloablSizeLength()))
            local_size =list(tuning.LocalSize(j) for j in range(tuning.LocalSizeLength()))

            if key not in autotuning_map:
                autotuning_map[key] = []
            
            autotuning_map[key].append((global_size, local_size))

        gemm_len = backend_info.GemmLength()

        for j in range(gemm_len):
            gemm = backend_info.Gemm(j)
            key = 'Xgemm_tune'
            gemm_size = list(gemm.GemmSize(j) for j in range(gemm.GemmSizeLength()))
            param_info =list(gemm.ParamInfo(j) for j in range(gemm.ParamInfoLength()))

            if key not in autotuning_map:
                autotuning_map[key] = []
            
            autotuning_map[key].append((gemm_size, param_info))

        opencl_tune_map[device_name] = autotuning_map

    # 生成 C++ 代码字符串
    cpp_code = "const std::map<std::string, std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>>> OpenCLTuneMap = {\n"
    
    for device, tuning_map in opencl_tune_map.items():
        cpp_code += f'    {{"{device}", {{\n'
        for key, size_pairs in tuning_map.items():
            cpp_code += f'        {{"{key}", {{\n'
            for sizes in size_pairs:
                cpp_code += f'            {{{{ {", ".join(map(str, sizes[0]))} }}, {{ {", ".join(map(str, sizes[1]))} }}}},\n'
            cpp_code += "        }},\n"    
        cpp_code += "    }},\n"
    
    cpp_code += "};\n"
    
    return cpp_code


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python import_cache.py <cache_file>")
        print("Example: python merge_cache.py mnn_cachefile.bin")
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        buffer = f.read()

    cpp_header_code = generate_cpp_header(buffer)

    # 将结果保存到头文件中
    with open('OpenCLTuneMap.hpp', 'w') as header_file:
        header_file.write("#include <map>\n#include <string>\n#include <vector>\n\nnamespace MNN { \n")
        header_file.write(cpp_header_code)
        header_file.write("\n}\n")

    print("C++ header file generated.")