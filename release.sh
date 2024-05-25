get_version() {
    version_header="./include/MNN/MNNDefine.h"
    version_major='x'
    version_minor='x'
    version_patch='x'

    # 读取 version_header 文件并提取版本信息
    while IFS='' read -r line || [[ -n "$line" ]]; do
        if echo "$line" | grep -q '#define MNN_VERSION_MAJOR'; then
            version_major=$(echo "$line" | awk '{print $3}')
        elif echo "$line" | grep -q '#define MNN_VERSION_MINOR'; then
            version_minor=$(echo "$line" | awk '{print $3}')
        elif echo "$line" | grep -q '#define MNN_VERSION_PATCH'; then
            version_patch=$(echo "$line" | awk '{print $3}')
        fi
    done < "$version_header"

    mnn_version="$version_major.$version_minor.$version_patch"
}


mnn() {
    echo 'build mnn release package.'
    # TODO
}

pymnn() {
    echo 'build pymnn release package.'
    get_version
    ./package_scripts/linux/build_whl.sh -v $mnn_version -o MNN-CPU/py_whl
    /opt/python/cp39-cp39/bin/python -m twine upload ./MNN-CPU/py_whl/*
}

case "$1" in
    mnn)
        mnn
        ;;
    pymnn)
        pymnn
        ;;
    *)
        $1
        echo $"Usage: $0 {mnn|pymnn}"
        exit 2
esac
exit $?
