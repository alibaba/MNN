set -e

usage() {
    echo "Usage: $0 -d wheel_path -v python_versions"
    echo -e "\t-d package files output directory"
    echo -e "\t-v MNN dist version"
    exit 1
}

while getopts "d:v:" opt; do
  case "$opt" in
    d ) path=$OPTARG ;;
    v ) IFS="," read -a python_versions <<< $OPTARG ;;
    * ) usage ;;
  esac
done

whls_path=$(realpath $path)
echo "whls pth is : " $whls_path
echo "python versions is : " $python_versions
source ~/miniconda/etc/profile.d/conda.sh
pushd pymnn/test
for env in $python_versions; do
    echo $env
    conda activate $env
    pip uninstall MNN_Internal -y
    if [ $env == "py27" ];
    then
        pip install numpy torch opencv-python==4.2.0.32
    else
        pip install numpy torch opencv-python
    fi 
    pip install -f $whls_path MNN-Internal
    python unit_test.py
    if [ $? -gt 0 ]; then
        conda deactivate
        popd
        echo "#### TEST FAILED ####"
        exit 1
    fi
    conda deactivate
done
echo "#### TEST SUCCESS ####"
popd
