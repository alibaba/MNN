# Copies from the files from Gitlab AliNN/MNN to Github MNN repo,
# and remove some internal files.
# This scripts assumes:
# 1. the current directory is the parent directory of "MNN"
# 2. the current directory contains the "GithubMNN" directory

SOURCE="MNN"
TARGET="GithubMNN"

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
ls | grep -v .git | xargs -I {} cp -af {} ../$TARGET
cp -f .gitignore ../$TARGET
popd > /dev/null

# reverting files
pushd $TARGET > /dev/null
# git clean -df
popd > /dev/null
