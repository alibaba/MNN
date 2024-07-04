import os
import sys
import re
major_py_ver = sys.version_info.major

def convert_string_to_hex_list(code_str):
    hex_list = []
    for i in range(len(code_str)):
        hex_ = hex(ord(code_str[i]))
        hex_list.append(hex_)
    return hex_list

def opencl_codegen():
    cl_kernel_dir = sys.argv[1]
    output_path = sys.argv[2]
    print("Generating OpenCL Kernels in "+cl_kernel_dir+" to "+output_path)
    if not os.path.exists(cl_kernel_dir):
        print(cl_kernel_dir + " doesn't exist!")

    opencl_code_maps = {}
    

#source model
    opencl_source_map = "#include <map> \n"
    opencl_source_map += "#include <string> \n"
    opencl_source_map += "#include <vector> \n"
    opencl_source_map += "#include <mutex> \n"
    opencl_source_map += "#include \"opencl_source_map.hpp\" \n"
    opencl_source_map += "namespace MNN { \n"
    opencl_source_map += "std::mutex gCLMutex;\n"
    
    opencl_source_hpp = "#include <map> \n"
    opencl_source_hpp += "#include <string> \n"
    opencl_source_hpp += "#include <vector> \n"
    opencl_source_hpp += "#include <mutex> \n"
    opencl_source_hpp += "namespace MNN { \n"

    opencl_source_map_hpp = "const std::map<std::string, const char*> OpenCLProgramMap = \n { \n"

    spaceReg = re.compile(' +')
    for file_name_all in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name_all)
        if file_path[-3:] == ".cl":
            with open(file_path, "r") as f:
                file_name = file_name_all[:-3]
                if file_name[-4:] == "_buf":
                    opencl_source_map += "#ifndef MNN_OPENCL_BUFFER_CLOSED\n"
                    opencl_source_hpp += "#ifndef MNN_OPENCL_BUFFER_CLOSED\n"
                    opencl_source_map_hpp += "#ifndef MNN_OPENCL_BUFFER_CLOSED\n"
                if file_name[-13:] == "_subgroup_buf":
                    opencl_source_map += "#ifdef MNN_SUPPORT_INTEL_SUBGROUP\n"
                    opencl_source_hpp += "#ifdef MNN_SUPPORT_INTEL_SUBGROUP\n"
                    opencl_source_map_hpp += "#ifdef MNN_SUPPORT_INTEL_SUBGROUP\n"
                opencl_source_hpp += "extern const char* " + file_name + ";\n"
                opencl_source_map += "const char* " + file_name + " = \n"
                opencl_source_map_hpp += "  { \"" + file_name + "\", " + file_name + " },\n"
                lines = f.read().split("\n")
                for l in lines:
                    if (len(l) < 1):
                        continue
                    if l.find('printf') >= 0:
                        l = l.replace('\"', '\\\"')
                        l = l.replace('\\n', '\\\\n')
                        opencl_source_map += "\""+l+"\"\n"
                    elif l.find('\\') >= 0:
                        l = l.replace('\\', '')
                        l = spaceReg.sub(' ', l)
                        opencl_source_map += "\""+l+"\""
                    else:
                        l = l + "\\n"
                        l = l.replace('\t', '')
                        l = spaceReg.sub(' ', l)
                        l = l.replace(', ', ',')
                        l = l.replace(' = ', '=')
                        l = l.replace(' + ', '+')
                        l = l.replace(' - ', '-')
                        l = l.replace(' * ', '*')
                        l = l.replace(' / ', '/')
                        l = l.replace(' < ', '<')
                        l = l.replace(' > ', '>')
                        opencl_source_map += "\""+l+"\"\n"
                opencl_source_map += ";\n"
                if file_name[-4:] == "_buf":
                    opencl_source_map += "#endif\n"
                    opencl_source_hpp += "#endif\n"
                    opencl_source_map_hpp += "#endif\n"
                if file_name[-13:] == "_subgroup_buf":
                    opencl_source_map += "#endif\n"
                    opencl_source_hpp += "#endif\n"
                    opencl_source_map_hpp += "#endif\n"
    opencl_source_map += "}\n"
    opencl_source_map_hpp += "};\n"
    opencl_source_map_hpp += "}\n"
    with open(output_path, "w") as w_file:
        w_file.write(opencl_source_map)
    with open("opencl_source_map.hpp", "w") as w_file:
        w_file.write(opencl_source_hpp)
        w_file.write(opencl_source_map_hpp)

    print("Generate OpenCL Source done !!! \n")

if __name__ == '__main__':
    opencl_codegen()
