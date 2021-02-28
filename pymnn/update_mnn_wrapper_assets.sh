set -e

usage() {
    echo "Usage: $0 -p python_version [-t]"
    echo -e "\t-p python versions in pyenv"
    echo -e "\t-t include train API wrapper"
    exit 1
}

while getopts "p:t" opt; do
  case "$opt" in
    p ) py_version=$OPTARG ;;
    t ) train_api=true ;;
    * ) usage ;;
  esac
done

rm -rf /tmp/mnn_py && mkdir -p /tmp/mnn_py
cp -r pip_package/MNN /tmp/mnn_py
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
pyenv global $py_version
python -c "import compileall; compileall.compile_dir('/tmp/mnn_py/MNN', force=True)"
find . -name *.py | xargs rm -rf
cd ..
zip -r MNN.zip MNN
popd

rm -f android/src/main/assets/MNN.zip
rm -rf iOS/MNNPyBridge/lib/MNN
cp /tmp/mnn_py/MNN.zip android/src/main/assets
cp -r /tmp/mnn_py/MNN iOS/MNNPyBridge/lib

rm -rf /tmp/mnn_py
