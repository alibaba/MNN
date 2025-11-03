import argparse
import os
import subprocess
import shutil

def generate_for_one_arch(exePath, qnnSDKPath, socId, hexagonArch, srcMNNPath, outputDir):
    """
    Calls the MNN2QNNModel tool for a single architecture.
    """
    print(f"Generating for socId: {socId}, hexagonArch: {hexagonArch}...")
    command = [
        exePath,
        qnnSDKPath,
        str(socId),
        str(hexagonArch),
        srcMNNPath,
        outputDir
    ]
    print(f"Executing: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        # print(result.stdout)
        print(f"Successfully generated for socId: {socId}, hexagonArch: {hexagonArch}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating for socId: {socId}, hexagonArch: {hexagonArch}")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error output:\n{e.stderr}")
        return False

def generate_for_all(exePath, qnnSDKPath, srcMNNPath, outputDir):
    """
    Iterates through all combinations and calls generate_for_one_arch for each.
    """
    combinations = [
        [36, '69'],
        [42, '69'],
        [43, '73'],
        [57, '75'],
        [69, '79']
    ]

    success_count = 0
    for socId, hexagonArch in combinations:
        if generate_for_one_arch(exePath, qnnSDKPath, socId, hexagonArch, srcMNNPath, outputDir):
            success_count += 1
    
    print(f"\nGeneration complete. {success_count}/{len(combinations)} architectures succeeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A tool to generate QNN offline caches for a specified model under different Qualcomm hardware architectures by calling 'MNN2QNNModel'.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--MNN2QNNModel_path", required=True, help="(Required) Path to the executable file 'MNN2QNNModel'.")
    parser.add_argument("--qnn_sdk_path", required=True, help="(Required) Path to the QNN SDK directory.")
    parser.add_argument("--src_mnn_path", required=True, help="(Required) Path to the source MNN model file.")
    parser.add_argument("--output_dir", default=".", help="(Optional) Directory to save the generated files. Default is the current working directory.")

    args = parser.parse_args()

    if not os.path.isfile(args.MNN2QNNModel_path):
        parser.error(f"MNN2QNNModel_path does not exist or is not a file: {args.MNN2QNNModel_path}")

    if not os.path.isdir(args.qnn_sdk_path):
        parser.error(f"qnn_sdk_path does not exist or is not a directory: {args.qnn_sdk_path}")

    if not os.path.isfile(args.src_mnn_path):
        parser.error(f"src_mnn_path does not exist or is not a file: {args.src_mnn_path}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    generate_for_all(args.MNN2QNNModel_path, args.qnn_sdk_path, args.src_mnn_path, args.output_dir)
