import json
import copy
import argparse
import os
import subprocess
import shutil # 导入 shutil 模块用于删除目录

def generate_all_configs(config_path, graph_name, qnn_sdk_root_path, src_model, executable_path, output_dir):
    """
    为每个组合创建子目录，生成配置文件，并调用C++可执行文件进行模型转换。
    """
    # --- 0. 准备工作 ---
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有生成的文件将被保存在主目录: '{output_dir}'")
    
    # 定义组合
    combinations = [
        [36, 'v69'],
        [42, 'v69'],
        [43, 'v73'],
        [57, 'v75'],
        [69, 'v79']
    ]

    # --- 1. 读取模板文件 ---
    htp_template_file = os.path.join(config_path, "htp_backend_extensions.json")
    context_template_file = os.path.join(config_path, "context_config.json")

    try:
        with open(htp_template_file, 'r', encoding='utf-8') as f:
            base_htp_data = json.load(f)
        print(f"成功读取模板文件 '{htp_template_file}'。")
        
        with open(context_template_file, 'r', encoding='utf-8') as f:
            base_context_data = json.load(f)
        print(f"成功读取模板文件 '{context_template_file}'。")
    except FileNotFoundError as e:
        print(f"错误：模板文件未找到。请确保 '{e.filename}' 存在于指定的路径中。")
        return
    except json.JSONDecodeError as e:
        print(f"错误：文件格式无效。请检查 {e.doc} 是否为有效的JSON。")
        return

    # --- 2. 遍历组合，生成文件并执行命令 ---
    for soc_id, dsp_arch in combinations:
        print(f"\n{'='*15} 处理组合: soc_id={soc_id}, dsp_arch={dsp_arch} {'='*15}")
        
        # --- 新增步骤: 为当前组合创建专用的子目录 ---
        new_graph_name = f"{graph_name}_{soc_id}_{dsp_arch}"
        graph_specific_dir = output_dir

        # --- Part A: 生成 htp_backend_extensions 文件 (路径更新) ---
        htp_config_data = copy.deepcopy(base_htp_data)
        try:
            htp_config_data["graphs"][0]["graph_names"] = [new_graph_name]
            htp_config_data["devices"][0]["soc_id"] = soc_id
            htp_config_data["devices"][0]["dsp_arch"] = dsp_arch
        except (IndexError, KeyError) as e:
            print(f"处理组合时出错: '{htp_template_file}' 结构不正确。错误: {e}")
            continue

        htp_output_filename = f"htp_backend_extensions_{soc_id}_{dsp_arch}.json"
        # 更新路径，使其指向新的子目录
        htp_output_filepath = os.path.join(graph_specific_dir, htp_output_filename)
        with open(htp_output_filepath, 'w', encoding='utf-8') as f:
            json.dump(htp_config_data, f, indent=4, ensure_ascii=False)
        print(f"-> 已生成配置文件: '{htp_output_filepath}'")
        
        # --- Part B: 生成 context_config 文件 (路径更新) ---
        context_config_data = copy.deepcopy(base_context_data)
        try:
            # 这里的 htp_output_filename 是相对路径，这是正确的，
            # 因为 context_config 和 htp_backend_extensions 在同一个目录中。
            context_config_data["backend_extensions"]["config_file_path"] = htp_output_filepath
            path_template = context_config_data["backend_extensions"]["shared_library_path"]
            new_lib_path = path_template.replace("{QNN_SDK_ROOT}", qnn_sdk_root_path)
            context_config_data["backend_extensions"]["shared_library_path"] = new_lib_path
        except KeyError as e:
            print(f"处理组合时出错: '{context_template_file}' 结构不正确，缺少键: {e}")
            continue
            
        context_output_filename = f"context_config_{soc_id}_{dsp_arch}.json"
        # 更新路径，使其指向新的子目录
        context_output_filepath = os.path.join(graph_specific_dir, context_output_filename)
        with open(context_output_filepath, 'w', encoding='utf-8') as f:
            json.dump(context_config_data, f, indent=4, ensure_ascii=False)
        print(f"-> 已生成关联文件: '{context_output_filepath}'")
        
        # --- Part C: 调用C++可执行命令 (路径更新) ---
        dst_model_filename = f"{graph_name}_{soc_id}_{dsp_arch}.mnn"
        # 更新路径，使其指向新的子目录
        dst_model_filepath = os.path.join(graph_specific_dir, dst_model_filename)
        
        graph_product_dir = os.path.join(graph_specific_dir, new_graph_name)
        os.makedirs(graph_product_dir, exist_ok=True)
        print(f"-> 已创建/确认子目录: '{graph_product_dir}'")

        command = [
            executable_path,
            src_model,
            dst_model_filepath,
            qnn_sdk_root_path,
            new_graph_name,
            context_output_filepath
        ]
        
        print("--> 准备执行命令...")
        print(f"    $ {' '.join(command)}")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("--> 命令执行成功!")
            # 即使成功，也打印 C++ 程序的输出，这对于查看警告等信息很有用
            if result.stdout:
                print("    --- C++程序输出 (stdout) ---")
                print(result.stdout.strip())
                print("    ------------------------------")

        except FileNotFoundError:
            print(f"!!! 命令执行失败: 可执行文件未找到 '{executable_path}'。请检查路径。")
            break # 如果可执行文件找不到，直接退出循环
        except subprocess.CalledProcessError as e:
            # 这是关键的修改部分
            print(f"!!! 命令执行失败 (返回码: {e.returncode})")
            
            # 检查并打印 C++ 程序在失败前产生的标准输出
            if e.stdout:
                print("    --- C++程序输出 (stdout) ---")
                print(e.stdout.strip())
                print("    ------------------------------")
            
            # 检查并打印 C++ 程序在失败前产生的标准错误（错误日志通常在这里）
            if e.stderr:
                print("    --- C++程序错误 (stderr) ---")
                print(e.stderr.strip())
                print("    ------------------------------")
        except Exception as e:
            print(f"!!! 执行期间发生未知错误: {e}")
        
        finally:
            # --- 步骤 3: 清理 ---
            # 检查目录是否存在，然后删除
            if os.path.exists(graph_product_dir):
                print(f"--> 清理临时文件和目录: '{graph_product_dir}'")
                shutil.rmtree(graph_product_dir)
            else:
                print("--> 无需清理，临时目录未创建。")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为多个组合创建子目录，生成QNN配置文件并调用模型转换工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (argparse部分保持完全不变) ...
    gen_group = parser.add_argument_group('文件生成参数')
    gen_group.add_argument("--config_path", required=True, help="[必需] 包含模板文件的目录路径。")
    gen_group.add_argument("--graph_name", required=True, help="[必需] 模型图的名称 (不含soc_id等后缀)。")
    gen_group.add_argument("--qnn_sdk_root_path", required=True, help="[必需] QNN SDK 的根路径。")
    
    exec_group = parser.add_argument_group('模型转换参数')
    exec_group.add_argument("--src_model", required=True, help="[必需] 源模型文件路径 (例如: my_model.mnn)。")
    exec_group.add_argument("--executable_path", required=True, help="[必需] C++模型转换可执行文件的路径。")
    exec_group.add_argument("--output_dir", default="./qnn_models", help="存放所有生成文件的输出目录 (默认: ./qnn_models)。")

    args = parser.parse_args()
    generate_all_configs(args.config_path, args.graph_name, args.qnn_sdk_root_path, args.src_model, args.executable_path, args.output_dir)
