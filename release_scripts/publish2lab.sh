# Copies from the files from Gitlab AliNN/AliNNPrivate to Gitlab AliNN/MNN repo,
# and remove some internal files.
# This scripts assumes:
# 1. the current directory is the parent directory of "AliNNPrivate"
# 2. the current directory contains the "MNN" directory

SOURCE="AliNNPrivate"
TARGET="MNN"

# check dirs
if [ ! -d $SOURCE ]; then
	echo "$SOURCE Not Found"
	exit -1
fi
if [ ! -d $TARGET ]; then
	echo "$TARGET Not Found"
	exit -1
fi

# remove files except .git in $TARGET
pushd $TARGET > /dev/null
ls | grep -v .git | xargs rm -rf
rm -f .gitignore
popd > /dev/null

# copy files from $SOURCE to $TARGET
pushd $SOURCE > /dev/null
# Remove gitignored and untracked files.
git clean -df

ls | grep -v .git | xargs -I {} cp -af {} ../$TARGET
cp -f .gitignore ../$TARGET
rm -rf ../$TARGET/release_scripts
rm -rf ../$TARGET/pymnn/android
rm -rf ../$TARGET/pymnn/iOS
rm -f ../$TARGET/pymnn/renameForAliNNPython.h
rm -f ../$TARGET/pymnn/src/private_define.h
rm -f ../$TARGET/pymnn/src/renameForAliNNPython.h
rm -f ../$TARGET/pymnn/MNNBridge.podspec
rm -f ../$TARGET/source/backend/hiai/3rdParty
popd > /dev/null

# reverting files
pushd $TARGET > /dev/null
git checkout -- benchmark/models/*.mnn
git checkout -- project/android/build.gradle
popd > /dev/null

# try re-build
pushd $TARGET > /dev/null

# MNN
rm -rf build
rm -rf schema/private
rm -rf schema/current

./schema/generate.sh
mkdir build && cd build
cmake .. -DMNN_BUILD_TEST=true -DMNN_BUILD_CONVERTER=true -DMNN_BUILD_QUANTOOLS=true
make -j4
./run_test.out

popd > /dev/null
