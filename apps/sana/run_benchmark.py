#!/usr/bin/env python3
"""
Sana Model Benchmark Script

Usage examples:
    # Run on host (Mac) with default settings
    python run_benchmark.py -i /path/to/images -r host -m /path/to/models

    # Run on Android device
    python run_benchmark.py -i /path/to/images -r android -b build_android -m sana_mnn_models_distill

    # Test multiple backends at once
    python run_benchmark.py -i /path/to/images -r host -m /path/to/models -k cpu metal -s 5 10

    # Single backend test
    python run_benchmark.py -i /path/to/image.jpg -r host -m /path/to/models -k cpu -s 10

    # Full options (Android)
    python run_benchmark.py -i /path/to/images -r android -o results -b build_android -m sana_mnn_models_distill -k cpu opencl -s 5 10 -p "your prompt"
"""

import subprocess
import os
import time
import csv
import shutil
import argparse
from pathlib import Path


VALID_BACKENDS = {
    "android": ["cpu", "opencl", "npu"],
    "host": ["cpu", "metal"]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sana Model Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input image file or directory containing images"
    )
    parser.add_argument(
        "-r", "--runner",
        choices=["android", "host"],
        default="android",
        help="Runner type: android (via adb) or host (local Mac) (default: android)"
    )
    parser.add_argument(
        "-o", "--output",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "-b", "--build-dir",
        help="Build directory (default: build_android for android, build_sana for host)"
    )
    parser.add_argument(
        "-m", "--model-dir",
        required=True,
        help="Model directory path"
    )
    parser.add_argument(
        "-k", "--backend",
        nargs="+",
        default=None,
        help="Backend type(s): cpu/opencl/npu for android, cpu/metal for host. Can specify multiple. (default: opencl for android, cpu for host)"
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="List of inference steps to test (default: 5 10 20)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="A beautiful scenery in Studio Ghibli style",
        help="Prompt for image generation"
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create zip archive of results"
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep existing output directory (do not clean)"
    )
    
    args = parser.parse_args()
    
    # Set defaults based on runner type
    if args.build_dir is None:
        args.build_dir = "build_android" if args.runner == "android" else "build_sana"
    if args.backend is None:
        args.backend = ["opencl"] if args.runner == "android" else ["cpu"]
    
    # Validate backend choices
    valid = VALID_BACKENDS[args.runner]
    for backend in args.backend:
        if backend not in valid:
            parser.error(f"Invalid backend '{backend}' for {args.runner}. Choose from: {', '.join(valid)}")
    
    return args


def get_image_files(input_path):
    """Get list of image files from input path (file or directory)."""
    input_path = Path(input_path)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    if input_path.is_file():
        if input_path.suffix.lower() in valid_extensions:
            return [input_path]
        else:
            raise ValueError(f"Invalid image file: {input_path}")
    elif input_path.is_dir():
        images = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        images.sort(key=lambda x: x.name)
        return images
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def parse_timer_output(stdout):
    """Parse timing metrics from command output."""
    metrics = {
        "Load LLM (ms)": "",
        "Init Diff (ms)": "",
        "LLM Infer (ms)": "",
        "Diff Infer (ms)": "",
        "Total (ms)": ""
    }
    
    import re
    
    # Parse [TIMER] format (from sana_diffusion_demo.cpp)
    timer_patterns = {
        r'\[TIMER\] Load LLM:\s*([\d.]+)': "Load LLM (ms)",
        r'\[TIMER\] Init Diffusion:\s*([\d.]+)': "Init Diff (ms)",
        r'\[TIMER\] LLM Inference:\s*([\d.]+)': "LLM Infer (ms)",
        r'\[TIMER\] Diffusion Inference:\s*([\d.]+)': "Diff Infer (ms)",
        r'\[TIMER\] Total:\s*([\d.]+)': "Total (ms)",
    }
    
    for pattern, metric_key in timer_patterns.items():
        match = re.search(pattern, stdout)
        if match:
            metrics[metric_key] = match.group(1)
    
    # Fallback: Parse "cost time" format from host script output (AUTOTIME macro)
    # e.g., "vae_encoder, 208, cost time: 2737.709961 ms"
    if not metrics["Load LLM (ms)"]:
        vae_enc_match = re.search(r'vae_encoder.*cost time:\s*([\d.]+)', stdout)
        if vae_enc_match:
            metrics["Load LLM (ms)"] = vae_enc_match.group(1)
    
    if not metrics["Diff Infer (ms)"]:
        # Sum up diffusion step times
        step_times = re.findall(r'Step \d+/\d+.*?run.*?cost time:\s*([\d.]+)', stdout, re.DOTALL)
        if step_times:
            total_step_time = sum(float(t) for t in step_times)
            metrics["Diff Infer (ms)"] = f"{total_step_time:.2f}"
    
    if not metrics["Total (ms)"]:
        vae_dec_match = re.search(r'vae_decoder.*cost time:\s*([\d.]+)', stdout)
        if vae_dec_match:
            # Use last run time as total if available
            total_match = re.findall(r'run, \d+, cost time:\s*([\d.]+)', stdout)
            if total_match:
                metrics["Total (ms)"] = total_match[-1]
    
    return metrics


def run_benchmark(args, backend, images, output_root):
    """Run benchmark tests for a single backend."""
    script_dir = Path(__file__).parent.resolve()
    
    # Convert paths to absolute
    build_dir = str(Path(args.build_dir).resolve())
    model_dir = str(Path(args.model_dir).resolve())
    
    results = []
    
    for img_path in images:
        img_name = img_path.name
        for steps in args.steps:
            print(f"Testing {img_name} with {steps} steps on {args.runner}/{backend}...")
            local_output = os.path.join(
                output_root,
                f"{img_path.stem}_{backend}_step_{steps}.jpg"
            )
            
            # Build command based on runner type
            if args.runner == "android":
                cmd = [
                    str(script_dir / "run_sana_on_android.sh"),
                    "-b", build_dir,
                    "-m", model_dir,
                    "-M", "img2img",
                    "-i", str(img_path),
                    "-o", str(Path(local_output).resolve()),
                    "-p", args.prompt,
                    "-k", backend,
                    "-s", str(steps)
                ]
            else:  # host
                cmd = [
                    str(script_dir / "run_sana_benchmark_host.sh"),
                    "-m", model_dir,
                    "-i", str(img_path),
                    "-b", backend,
                    "-s", str(steps),
                    "-M", "img2img"
                ]
            
            start_time = time.time()
            try:
                process = subprocess.run(cmd, capture_output=True, text=True)
                end_time = time.time()
                
                # Combine stdout and stderr for parsing
                output_text = process.stdout + "\n" + process.stderr
                metrics = parse_timer_output(output_text)
                
                result = {
                    "Image": img_name,
                    "Runner": args.runner,
                    "Backend": backend,
                    "Steps": steps,
                    **metrics,
                    "Total Script Time (s)": round(end_time - start_time, 2)
                }
                results.append(result)
                
                if process.returncode != 0:
                    print(f"  Warning: Command returned non-zero exit code ({process.returncode})")
                    if process.stderr:
                        # Print last few lines of stderr
                        stderr_lines = process.stderr.strip().split('\n')[-3:]
                        for line in stderr_lines:
                            print(f"  stderr: {line[:100]}")
                        
            except Exception as e:
                print(f"  Error: {e}")
    
    return results


def save_results(results, output_root, create_zip=True):
    """Save results to CSV and optionally create zip archive."""
    if not results:
        print("No results to save.")
        return
    
    # Save CSV
    csv_path = os.path.join(output_root, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    headers = list(results[0].keys())
    print(" | ".join(f"{h[:15]:>15}" for h in headers))
    print("-" * 80)
    for r in results:
        print(" | ".join(f"{str(v)[:15]:>15}" for v in r.values()))
    
    # Create zip
    if create_zip:
        zip_name = f"{output_root}_results"
        shutil.make_archive(zip_name, 'zip', output_root)
        print(f"\nResults archived to: {zip_name}.zip")


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Sana Model Benchmark")
    print("=" * 50)
    
    # Get image files
    try:
        images = get_image_files(args.input)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not images:
        print(f"No valid images found in: {args.input}")
        return
    
    print(f"Found {len(images)} image(s) to test")
    print(f"Runner: {args.runner}")
    print(f"Backends: {', '.join(args.backend)}")
    print(f"Steps: {args.steps}")
    print(f"Model dir: {args.model_dir}")
    print("-" * 50)
    
    # Setup output directory
    backends_str = "_".join(args.backend)
    output_root = f"{args.output}_{args.runner}_{backends_str}"
    if not args.keep_output and os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    
    # Run benchmarks for each backend
    all_results = []
    for backend in args.backend:
        print(f"\n{'='*50}")
        print(f"Testing backend: {backend}")
        print(f"{'='*50}")
        results = run_benchmark(args, backend, images, output_root)
        all_results.extend(results)
    
    save_results(all_results, output_root, create_zip=not args.no_zip)
    
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
