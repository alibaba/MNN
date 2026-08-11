import subprocess
import os
import time
import csv
import shutil

new_image_path = "/Users/songjinde/git/days4/20260121164805.jpg"
output_root = "benchmark_results_new_image"
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root, exist_ok=True)

build_dir = "build_android"
model_dir = "sana_mnn_models_distill"
prompt = "A beautiful scenery in Studio Ghibli style"
steps_list = [5, 10, 20]
backends = ["opencl", "cpu"]

results = []

for backend in backends:
    for steps in steps_list:
        print(f"Testing new image with {backend} backend and {steps} steps...")
        img_name = os.path.basename(new_image_path)
        local_output = os.path.join(output_root, f"{os.path.splitext(img_name)[0]}_{backend}_step_{steps}.jpg")
        
        cmd = [
            "./run_sana_on_android.sh",
            "-b", build_dir,
            "-m", model_dir,
            "-i", new_image_path,
            "-o", local_output,
            "-p", prompt,
            "-k", backend,
            "-s", str(steps)
        ]
        
        start_time = time.time()
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            stdout = process.stdout
            end_time = time.time()
            
            # Parse metrics
            load_llm = ""
            init_diff = ""
            infer_llm = ""
            load_diff = ""
            infer_diff = ""
            
            for line in stdout.split('\n'):
                if "[TIMER] Load LLM:" in line:
                    load_llm = line.split(":")[-1].strip().split()[0]
                elif "[TIMER] Init Diffusion:" in line:
                    init_diff = line.split(":")[-1].strip().split()[0]
                elif "[TIMER] LLM Inference:" in line:
                    infer_llm = line.split(":")[-1].strip().split()[0]
                elif "[TIMER] Load Diffusion Weights:" in line:
                    load_diff = line.split(":")[-1].strip().split()[0]
                elif "[TIMER] Diffusion Inference:" in line:
                    infer_diff = line.split(":")[-1].strip().split()[0]
            
            results.append({
                "Backend": backend,
                "Steps": steps,
                "Load LLM (ms)": load_llm,
                "Init Diffusion (ms)": init_diff,
                "LLM Inference (ms)": infer_llm,
                "Load Diff Weights (ms)": load_diff,
                "Diff Inference (ms)": infer_diff,
                "Total Script Time (s)": round(end_time - start_time, 2)
            })
            
        except Exception as e:
            print(f"Error testing with {backend} and {steps} steps: {e}")

if results:
    csv_path = os.path.join(output_root, "new_image_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("New image results saved to benchmark_results_new_image/")
else:
    print("No results generated.")
