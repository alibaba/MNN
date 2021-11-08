## scp package_scripts/linux/build.sh mnnteam@30.6.159.68:/mnt/partition4/CI/scripts
# scp ~/.ssh/id_rsa* mnnteam@30.6.159.68:/mnt/partition4/CI
# ssh mnnteam@30.6.159.68
# docker run --name CI_tmp --rm -it -v /mnt/partition4/CI:/mnt reg.docker.alibaba-inc.com/shuhui/manylinux_2014 bash /mnt/scripts/build.sh -r git@gitlab.alibaba-inc.com:AliNN/AliNNPrivate.git
# docker run --name CI_tmp --rm -it -v /mnt/partition4/CI:/mnt reg.docker.alibaba-inc.com/shuhui/manylinux_2014 bash /mnt/scripts/build.sh -r git@gitlab.alibaba-inc.com:AliNN/MNN.git
# docker run --name CI_tmp --rm -it -v /mnt/partition4/CI:/mnt reg.docker.alibaba-inc.com/shuhui/manylinux_2014 bash /mnt/scripts/build.sh -r git@github.com:alibaba/MNN.git

set -e

usage() {
    echo "Usage: $0 -r code_repo"
    echo -e "\t-r code repository"
    exit 1
}

while getopts 'r:' opt; do
  case "$opt" in
    r ) CODE_REPO=$OPTARG ;;
    h|? ) usage ;;
  esac
done

yes | cp /mnt/id_rsa* ~/.ssh 2>/dev/null
cd /root
git clone $CODE_REPO MNN && cd MNN
mkdir MNN
cp -r include/* MNN
./package_scripts/linux/build_lib.sh -o MNN-CPU/lib
# ./package_scripts/linux/build_tools.sh -o MNN-CPU/tools
./package_scripts/linux/build_whl.sh -o MNN-CPU/py_whl
# ./package_scripts/linux/build_bridge.sh -o MNN-CPU/py_bridge
