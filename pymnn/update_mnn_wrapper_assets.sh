#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -p python_version -v mnn_version [-t]"
    echo -e "\t-p python versions in pyenv [only support 2.x]"
    echo -t "\t-t include train API wrapper"
    echo -c "\t-c check pyc need update"
    exit 1
}

while getopts "p:v:t:c" opt; do
  case "$opt" in
    p ) py_version=$OPTARG ;;
    t ) train_api=true ;;
    c ) check=true;;
    * ) usage ;;
  esac
done

cmdExist() {
    which $1 > /dev/null 2>&1
    return $?
}

rm -rf /tmp/mnn_py && mkdir -p /tmp/mnn_py
cp -r pip_package/MNN /tmp/mnn_py
# remove comments
python3 rm_comments.py /tmp/mnn_py/MNN
pushd /tmp/mnn_py/MNN

rm -rf tools
cat __init__.py | sed '/from . import tools/d' > __init__.py.tmp
mv __init__.py.tmp __init__.py

if [ -z $train_api ]; then
    rm -rf data optim
    cat __init__.py | sed '/from . import data/d' | sed '/from . import optim/d' > __init__.py.tmp
    mv __init__.py.tmp __init__.py
fi

find . -name __pycache__ | xargs rm -rf
if cmdExist pyenv; then
    pyenv global $py_version
fi
python -c "import compileall; compileall.compile_dir('/tmp/mnn_py/MNN', force=True)"
find . -name "*.py" | xargs rm -rf
cd ..
zip -r MNN.zip MNN
popd

# update wrapper assets from $1 to $2 when pyc (WITHOUT METADATA) is not same
should_update () {
    pushd $1
    pyc_files_1=(`find MNN -name *.pyc | sort`)
    popd
    pushd $2
    pyc_files_2=(`find MNN -name *.pyc | sort`)
    popd
    if [ ${#pyc_files_1[@]} -ne ${#pyc_files_2[@]} ]; then
        return 0
    fi
    for ((i=0;i<${#pyc_files_1[@]};i++)); do
        if [ ${pyc_files_1[i]} != ${pyc_files_2[i]} ]; then
            return 0
        fi
        pyc_file=${pyc_files_1[i]}
        if cmdExist md5sum; then
            sum_old=`tail -c +8 $2/$pyc_file | md5sum | awk '{print $1}'`
            sum_new=`tail -c +8 $1/$pyc_file | md5sum | awk '{print $1}'`
        else
            sum_old=`tail -c +8 $2/$pyc_file | md5 | awk '{print $1}'`
            sum_new=`tail -c +8 $1/$pyc_file | md5 | awk '{print $1}'`
        fi
        if [ $sum_old != $sum_new ]; then
            return 0
        fi
    done
    return 1
}

if [ $check ]; then
    if should_update /tmp/mnn_py iOS/MNNPyBridge/lib; then
        echo 'need update'
        exit 1
    else
        echo 'need not update'
        exit 0
    fi
fi

if should_update /tmp/mnn_py iOS/MNNPyBridge/lib; then
    rm -f android/src/main/assets/MNN.zip
    rm -rf iOS/MNNPyBridge/lib/MNN
    cp /tmp/mnn_py/MNN.zip android/src/main/assets
    cp -r /tmp/mnn_py/MNN iOS/MNNPyBridge/lib
fi

rm -rf /tmp/mnn_py
