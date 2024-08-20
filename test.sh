# test script for MNN-Release
#
# 0. arg = local: [ test for your local build ]
#       1. unit-test;
#       2. model-test;
#       3. onnx convert test
#       4. tf convert test
#       5. tflite convert test
#       6. torch convert test
#       7. ptq test
#       8. pymnn test
#
# 1. arg = linux: [ all test on linux with coverage ]
#       0. static check (if source change)
#       1. pyc check (if *.py change)
#       2. build for linux;
#       3. unit-test;
#       4. model-test;
#       5. onnx convert test
#       6. tf convert test
#       7. tflite convert test
#       8. torch convert test
#       9. ptq test
#      10. pymnn test (if pymnn change)
#      11. opencv test (if opencv change)
#      12. convert-report;
#
# 2. arg = android: [ simple test on android ]
#       1. build Android with static_stl
#       2. build Android arm64
#       3. unit-test for Android arm64
#       4. build Android arm32
#       5. unit-test for Android arm32

# 0. build for android
USER_NAME=`whoami`
USER_HOME="$(echo -n $(bash -c "cd ~${USER_NAME} && pwd"))"

# detect change
SOURCE_CHANGE=$(git show --name-only | grep -E "^source/(internal|backend|core|common|cv|geometry|math|plugin|shape|utils)/.*\.(cpp|cc|c|hpp)$" | \
                grep -Ev "aliyun-log-c-sdk|hiai|tensorrt|Backend|FunctionDispatcher|ThreadPool")
PYMNN_CHANGE=$(git show --name-only | grep -E "^pymnn/.*\.(cpp|cc|c|h|hpp|py)$")
PY_CHANGE=$(git show --name-only | grep -E "^pymnn/pip_package/MNN/.*\.(py)$")
OPENCV_CHANGE=$(git show --name-only | grep -E "^tools/cv/.*\.(cpp|cc|c|h|hpp)$")
# OPENCL_CHANGE=$(git show --name-only | grep -E "^source/backend/opencl/.*\.(cpp|cc|c|h|hpp)$")
OPENCL_CHANGE=true
failed() {
    printf "TEST_NAME_EXCEPTION: Exception\nTEST_CASE_AMOUNT_EXCEPTION: {\"blocked\":0,\"failed\":1,\"passed\":0,\"skipped\":0}\n"
    exit 1
}

#############################################################################################
#                                                                                           #
#                                  Linux Test Functions                                     #
#                                                                                           #
#############################################################################################
doc_check() {
    echo 'doc_check'
    # 1. CHECK CMakeLists.txt:
    cmake_files=$(find tools source demo test benchmark  -name "CMakeLists.txt")
    cmake_files="$cmake_files CMakeLists.txt"
    macros=''
    executables=''
    for cmake_file in $cmake_files
    do
        executables="$executables $(cat $cmake_file | grep -oE "add_executable\((.+) " | awk '{print $1}' | awk -F "(" '{print $2}')"
        macros="$macros $(cat $cmake_file | grep -oE "option\((.+) " | awk '{print $1}' | awk -F "(" '{print $2}')"
    done
    # 1.1 check all macro
    for macro in $macros
    do
        if [ $(grep -c $macro ./docs/compile/cmake.md) -le 0 ]; then
            echo 'DOC CHECK FAILED:' $macro 'not in ./docs/compile/cmake.md'
            failed
        fi
    done
    # 1.2 check executable
    for executable in $executables
    do
        if [ $(grep -c $executable ./docs/compile/tools.md) -le 0 ]; then
            echo 'DOC CHECK FAILED:' $executable 'not in ./docs/compile/tools.md'
            failed
        fi
    done
    # 2. CHECK Pymnn API:
    # 2.1 check cv api
    cv_apis=$(cat pymnn/src/cv.h | grep -oE "        .+, \".+\"" | awk '{ print $1 }' | awk -F ',' '{ print $1 }')
    cv_apis="$cv_apis $(cat pymnn/pip_package/MNN/cv/__init__.py | grep -oE "def .+\(" | awk '{ print $2 }' | awk -F '(' '{print $1}' | grep -v "__")"
    for cv_api in $cv_apis
    do
        if [ $(grep -c $cv_api ./docs/pymnn/cv.md) -le 0 ]; then
            echo 'DOC CHECK FAILED:' $cv_api 'not in ./docs/pymnn/cv.md'
            failed
        fi
    done
    # 2.2 check numpy api
    # np_apis=$(cat pymnn/pip_package/MNN/numpy/__init__.py | grep -oE "def .+\(" | grep -v "__" | awk '{ print $2 }' | awk -F '(' '{print $1}')
    # for np_api in $np_apis
    # do
    #     if [ $(grep -c $np_api ./docs/pymnn/numpy.md) -le 0 ]; then
    #         echo 'DOC CHECK FAILED:' $np_api 'not in ./docs/pymnn/numpy.md'
    #         # failed
    #     fi
    # done
    # 2.3 check expr api
    expr_apis=$(cat pymnn/src/expr.h | grep -oE "        [a-z_]+, \"" | awk '{ print $1 }' | awk -F ',' '{ print $1 }')
    for expr_api in $expr_apis
    do
        if [ $(grep -c $expr_api ./docs/pymnn/expr.md) -le 0 ]; then
            echo 'DOC CHECK FAILED:' $expr_api 'not in ./docs/pymnn/expr.md'
            # failed
        fi
    done
    # 3. CHECK C++ API:
    # 3.1 check Interpreter
    # 3.2 check Tensor
}

py_check() {
    if [ -z "$PY_CHANGE" ]; then
        return
    fi
    pushd pymnn
    ./update_mnn_wrapper_assets.sh -c
    pyc_check_wrong=$[$? > 0]
    printf "TEST_NAME_PYC_CHECK: pyc资源文件校验\nTEST_CASE_AMOUNT_PYC_CHECK: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $pyc_check_wrong $[1 - $pyc_check_wrong]
    if [ $pyc_check_wrong -ne 0 ]; then
        echo '### pyc资源文件校验失败，测试终止！'
        failed
    fi
    popd
}

static_check() {
    if [ -z "$SOURCE_CHANGE" ]; then
        return
    fi
    cppcheck --error-exitcode=1 --language=c++ --std=c++14 --addon=tools/script/mnn_rules.py $SOURCE_CHANGE 1> /dev/null
    static_check_wrong=$[$? > 0]
    printf "TEST_NAME_STATIC_CHECK: cppcheck静态分析\nTEST_CASE_AMOUNT_STATIC_CHECK: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $static_check_wrong $[1 - $static_check_wrong]
    if [ $static_check_wrong -ne 0 ]; then
        echo '### cppcheck静态分析失败，测试终止！'
        failed
    fi
}

android_static_build() {
    BASH_FILE="$USER_HOME/.zshrc"
    if [ -f "$BASH_FILE" ]; then
        source $BASH_FILE
    fi
    if [ ! $ANDROID_NDK ] || [ ! -d $ANDROID_NDK ]; then
        export ANDROID_NDK="$USER_HOME/android-ndk-r21"
    fi
    mkdir android_build
    pushd android_build
    cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DMNN_INTERNAL=ON \
    -DMNN_USE_LOGCAT=false \
    -DMNN_BUILD_BENCHMARK=ON \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DMNN_OPENGL=true \
    -DMNN_BUILD_TRAIN=true \
    -DMNN_VULKAN=true \
    -DMNN_OPENCL=true \
    -DMNN_SUPPORT_BF16=true \
    -DMNN_OPENCL=true -DMNN_ARM82=true \
    -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3
    make -j16
    android_build_wrong=$[$? > 0]
    printf "TEST_NAME_ANDROID_STATIC: AndroidStatic编译测试\nTEST_CASE_AMOUNT_ANDROID_STATIC: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $android_build_wrong $[1 - $android_build_wrong]
    if [ $android_build_wrong -ne 0 ]; then
        echo '### AndroidStatic编译失败，测试终止！'
        failed
    fi
    popd

    mkdir android_build_32
    pushd android_build_32
    cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="armeabi-v7a" \
    -DANDROID_STL=c++_shared \
    -DMNN_USE_LOGCAT=false \
    -DMNN_BUILD_BENCHMARK=ON \
    -DMNN_INTERNAL=ON \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DMNN_OPENGL=true \
    -DMNN_BUILD_TRAIN=true \
    -DMNN_VULKAN=true \
    -DMNN_OPENCL=true \
    -DMNN_BUILD_MINI=true \
    -DMNN_SUPPORT_BF16=true \
    -DMNN_OPENCL=true\
    -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.
    make -j16
    android_build_wrong=$[$? > 0]
    printf "TEST_NAME_ANDROID_32: Android 32-Mini 编译测试\nTEST_CASE_AMOUNT_ANDROID_32: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $android_build_wrong $[1 - $android_build_wrong]
    if [ $android_build_wrong -ne 0 ]; then
        echo '### Android编译失败，测试终止！'
        failed
    fi
    popd
}

linux_build() {
    if [ $# -gt 0 ]; then
        COVERAGE=ON
    else
        COVERAGE=OFF
    fi

    mkdir build_non_sse
    pushd build_non_sse
    cmake .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DMNN_USE_SSE=OFF && make -j16

    linux_build_wrong=$[$? > 0]
    popd

    mkdir build
    pushd build
    # copy libtorch avoid wget, speed up ci build
    cp ~/libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip .
    cmake .. \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TEST=ON \
        -DMNN_CUDA=ON \
        -DMNN_OPENCL=ON \
        -DMNN_BUILD_QUANTOOLS=ON \
        -DMNN_BUILD_DEMO=ON \
        -DMNN_BUILD_TRAIN=ON \
        -DMNN_BUILD_CONVERTER=ON \
        -DMNN_BUILD_TORCH=ON \
        -DMNN_BUILD_OPENCV=ON \
        -DMNN_LOW_MEMORY=ON \
        -DMNN_IMGCODECS=ON \
        -DMNN_ENABLE_COVERAGE=$COVERAGE
    make -j16

    linux_build_wrong+=$[$? > 0]
    printf "TEST_NAME_LINUX: Linux编译测试\nTEST_CASE_AMOUNT_LINUX: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $linux_build_wrong $[2 - $linux_build_wrong]
    if [ $linux_build_wrong -ne 0 ]; then
        echo '### Linux编译失败，测试终止！'
        failed
    fi

    # Don't remove this! It turn off MNN_CUDA and MNN_TENSORRT in build, workaround some bug in PTQTest
    cmake .. -DMNN_CUDA=OFF -DMNN_TENSORRT=OFF && make -j16
}

unit_test() {
    ./run_test.out
    if [ $? -ne 0 ]; then
        echo '### 单元测试失败，测试终止！'
        failed
    fi

    ./run_test.out op 0 0 4
    if [ $? -ne 0 ]; then
        echo '### 多线程单元测试失败，测试终止！'
        failed
    fi
    if [ "$OPENCL_CHANGE" ]; then
        ./run_test.out op 3 1 4
        if [ $? -ne 0 ]; then
            echo '### OpenCL单元测试失败，测试终止！'
            failed
        fi
    fi
}

model_test() {
    ../tools/script/modelTest.py ~/AliNNModel 0 0.002
    if [ $? -ne 0 ]; then
        echo '### 模型测试失败，测试终止！'
        failed
    fi

    ../tools/script/modelTest.py ~/AliNNModel 0 0.002 0 1
    if [ $? -ne 0 ]; then
        echo '### 静态模型测试失败，测试终止！'
        failed
    fi

    if [ "$OPENCL_CHANGE" ]; then
        ../tools/script/modelTest.py ~/AliNNModel 3 0.002 1
        if [ $? -ne 0 ]; then
            echo '### OpenCL模型测试失败，测试终止！'
            failed
        fi
    fi
}

onnx_convert_test() {
    ../tools/script/convertOnnxTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### ONNXConvert测试失败，测试终止！'
        failed
    fi
    if [ -f ~/AliNNModel/TestOnnx/ops/run.py ]; then
        ~/AliNNModel/TestOnnx/ops/run.py --mnndir $(pwd) --aone-mode
        if [ $? -ne 0 ]; then
            echo '### Onnx单线程单元测试失败，测试终止！'
            failed
        fi
        #~/AliNNModel/TestOnnx/ops/run.py --mnndir $(pwd) --aone-mode --thread_num 2
        #if [ $? -ne 0 ]; then
        #    echo '### ONNX多线程单元测试失败，测试终止！'
        #    failed
        #fi
    fi
}

tf_convert_test() {
    ../tools/script/convertTfTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TFConvert测试失败，测试终止！'
        failed
    fi
}

tflite_convert_test() {
    ../tools/script/convertTfliteTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TFLITEConvert测试失败，测试终止！'
        failed
    fi
}

torch_convert_test() {
    ../tools/script/convertTorchTest.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### TORCHConvert测试失败，测试终止！'
        failed
    fi
}

ptq_test() {
    ../tools/script/testPTQ.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### PTQ测试失败，测试终止！'
        failed
    fi
}

pymnn_test() {
    if [ -z "$PYMNN_CHANGE" ]; then
        return
    fi
    popd
    pushd pymnn
    # 1. build pymnn
    pushd pip_package
    python3 build_deps.py
    # uninstall original MNN
    pip uninstall --yes MNN MNN-Internal
    python3 setup.py install --version 1.0 --install-lib=/usr/lib/python3/dist-packages
    pymnn_build_wrong=$[$? > 0]
    printf "TEST_NAME_PYMNN_BUILD: PYMNN编译测试\nTEST_CASE_AMOUNT_PYMNN_BUILD: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $pymnn_build_wrong $[1 - $pymnn_build_wrong]
    if [ $pymnn_build_wrong -ne 0 ]; then
        echo '### PYMNN编译失败，测试终止！'
        failed
    fi
    popd
    # 2. unit test
    pushd test
    python3 unit_test.py
    if [ $? -ne 0 ]; then
        echo '### PYMNN单元测试失败，测试终止！'
        failed
    fi
    # 3. model test
    python3 model_test.py ~/AliNNModel
    if [ $? -ne 0 ]; then
        echo '### PYMNN模型测试失败，测试终止！'
        failed
    fi
    # 4. train test
    ./train_test.sh
    # 5. quant test
    python3 ../examples/MNNQuant/test_mnn_offline_quant.py \
            --mnn_model ~/AliNNModel/TestQuant/mobilenet_v2_tfpb_train_withBN.mnn \
            --quant_imgs ~/AliNNModel/TestQuant/quant_imgs \
            --quant_model ./quant_model.mnn
    rm ./quant_model.mnn
    quant_wrong=$[$? > 0]
    printf "TEST_NAME_QUANT_TEST: pymnn量化测试\nTEST_CASE_AMOUNT_QUANT_TEST: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $quant_wrong $[1 - $quant_wrong]
    # 6. uninstall pymnn
    pip uninstall --yes MNN-Internal
    popd
    popd
    pushd build
}

opencv_test() {
    if [ -z "$OPENCV_CHANGE" ]; then
        return
    fi
    # 1. build opencv-test
    cmake -DMNN_OPENCV_TEST=ON ..
    make -j8
    opencv_build_wrong=$[$? > 0]
    printf "TEST_NAME_OPENCV_BUILD: OPENCV编译测试\nTEST_CASE_AMOUNT_OPENCV_BUILD: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $opencv_build_wrong $[1 - $opencv_build_wrong]
    if [ $opencv_build_wrong -ne 0 ]; then
        echo '### OPENCV编译失败，测试终止！'
        failed
    fi
    # 2. run opencv unit test
    ./opencv_test
    if [ $? -gt 0 ]; then
        echo '### OPENCV单元测试失败，测试终止！'
        failed
    fi
}

llm_test() {
    # 1. build llm with low memory
    cmake -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON ..
    make -j8
    llm_build_wrong=$[$? > 0]
    printf "TEST_NAME_LLM_BUILD: LLM编译测试\nTEST_CASE_AMOUNT_LLM_BUILD: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $llm_build_wrong $[1 - $llm_build_wrong]
    if [ $llm_build_wrong -ne 0 ]; then
        echo '### LLM编译失败，测试终止！'
        failed
    fi
    # 2. run llm model test
    ./llm_demo ~/AliNNModel/qwen1.5-0.5b-int4/config.json ~/AliNNModel/qwen1.5-0.5b-int4/prompt.txt
    if [ $? -gt 0 ]; then
        echo '### LLM模型测试失败，测试终止！'
        failed
    fi
}

coverage_init() {
    popd
    lcov -c -i -d ./ -o init.info
    pushd build
}

coverage_report() {
    popd
    cover_report_dir="../../../../CoverageReport"
    lcov -c -d ./ -o cover.info
    lcov -a init.info -a cover.info -o total.info
    lcov --remove total.info \
    '*/usr/include/*' '*/usr/lib/*' '*/usr/lib64/*' '*/usr/local/*'  \
    '*/3rd_party/*' '*/build/*' '*/schema/*' '*/test/*' '/tmp/*' \
    '*/demo/*' '*/tools/cpp/*' '*/tools/train/*' '*/source/backend/cuda/*' \
    -o final.info
    commitId=$(git log | head -n1 | awk '{print $2}')
    genhtml -o cover_report --legend --title "MNN Coverage Report [commit SHA1:${commitId}]" --prefix=`pwd` final.info
    coverage_wrong=$[$? > 0]
    printf "TEST_NAME_COVERAGE: 代码覆盖率(点击\"通过\"查看报告)\nTEST_CASE_AMOUNT_COVERAGE: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $coverage_wrong $[1 - $coverage_wrong]
    if [ $coverage_wrong -ne 0 ]; then
        echo '### 代码覆盖率生成失败，测试终止！'
        failed
    else
        hostIp=$(cat .aoneci.yml | grep host -m 1 | awk '{print $2}')
        testId=$(pwd | awk -F "/" '{print $(NF-1)}')
        mv cover_report $cover_report_dir/$testId
        echo "TEST_REPORT_COVERAGE: http://$hostIp/$testId"
    fi
    # clean test dir
    cd ../.. && rm -rf $testId
}

#############################################################################################
#                                                                                           #
#                                  Android Test Functions                                   #
#                                                                                           #
#############################################################################################
android_unit_test() {
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out all 0 0 1 $1"
    if [ $? -ne 0 ]; then
        echo '### Android单元测试失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op 0 0 4 multi$1"
    if [ $? -ne 0 ]; then
        echo '### Android单元测试多线程失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/convolution 0 2 4 fp16multi$1"
    if [ $? -ne 0 ]; then
        echo '### Android单元测试卷积FP16多线程失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/col2im 0 2 4 fp16col2im$1"
    if [ $? -ne 0 ]; then
        echo '### Android单元测试FP16-col2im多线程失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/R 0 2 4 fp16roipooling$1"
    if [ $? -ne 0 ]; then
        echo '### Android单元测试FP16-roipooling多线程失败，测试终止！'
        failed
    fi
    if [ "$OPENCL_CHANGE" ]; then
        adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op 3 1 4 $1"
        if [ $? -ne 0 ]; then
            echo '### Android单元测试OpenCL失败，测试终止！'
            failed
        fi
    fi
}
android_model_test() {
    fail_num=0
    pass_num=0
    fail_cl_num=0
    pass_cl_num=0
    models=`ls ~/AliNNModel/OpTestResource/`
    for model in $models
    do
        adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModel.out ../AliNNModel/OpTestResource/$model/temp.bin ../AliNNModel/OpTestResource/$model/input_0.txt ../AliNNModel/OpTestResource/$model/output_0.txt 0 0.002"
        if [ $? -ne 0 ]; then
            fail_num=$[$fail_num+1]
        else
            pass_num=$[$pass_num+1]
        fi
        if [ "$OPENCL_CHANGE" ]; then
            adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModel.out ../AliNNModel/OpTestResource/$model/temp.bin ../AliNNModel/OpTestResource/$model/input_0.txt ../AliNNModel/OpTestResource/$model/output_0.txt 3 0.002 1"
            if [ $? -ne 0 ]; then
                fail_cl_num=$[$fail_cl_num+1]
            else
                pass_cl_num=$[$pass_cl_num+1]
            fi
        fi
    done

    models=`ls ~/AliNNModel/TestResource/`
    for model in $models
    do
        if [ $model == 'mobilenetv1quan' ]; then
            adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModel.out ../AliNNModel/TestResource/$model/temp.bin ../AliNNModel/TestResource/$model/input_0.txt ../AliNNModel/TestResource/$model/output.txt 0 0.1"
        else
            adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModel.out ../AliNNModel/TestResource/$model/temp.bin ../AliNNModel/TestResource/$model/input_0.txt ../AliNNModel/TestResource/$model/output.txt 0 0.002"
        fi
        if [ $? -ne 0 ]; then
            fail_num=$[$fail_num+1]
        else
            pass_num=$[$pass_num+1]
        fi
        if [ "$OPENCL_CHANGE" ]; then
        if [ $model == 'mobilenetv1quan' ]; then
            adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModel.out ../AliNNModel/TestResource/$model/temp.bin ../AliNNModel/TestResource/$model/input_0.txt ../AliNNModel/TestResource/$model/output.txt 3 0.1 1"
        else 
            adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModel.out ../AliNNModel/TestResource/$model/temp.bin ../AliNNModel/TestResource/$model/input_0.txt ../AliNNModel/TestResource/$model/output.txt 3 0.002 1"
        fi    
            if [ $? -ne 0 ]; then
                fail_cl_num=$[$fail_cl_num+1]
            else
                pass_cl_num=$[$pass_cl_num+1]
            fi
        fi
    done

    models=`ls ~/AliNNModel/TestWithDescribe/`
    for model in $models
    do
        adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModelWithDescribe.out ../AliNNModel/TestWithDescribe/$model/temp.bin ../AliNNModel/TestWithDescribe/$model/config.txt 0 0.002"
        if [ $? -ne 0 ]; then
            fail_num=$[$fail_num+1]
        else
            pass_num=$[$pass_num+1]
        fi
        if [ "$OPENCL_CHANGE" ]; then
            adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./testModelWithDescribe.out ../AliNNModel/TestWithDescribe/$model/temp.bin ../AliNNModel/TestWithDescribe/$model/config.txt 3 0.002 1"
            if [ $? -ne 0 ]; then
                fail_cl_num=$[$fail_cl_num+1]
            else
                pass_cl_num=$[$pass_cl_num+1]
            fi
        fi
    done
    printf "TEST_NAME_ANDROID_MODEL_TEST_$1: Android_$1模型测试\nTEST_CASE_AMOUNT_ANDROID_MODEL_TEST_$1: {\"blocked\":0,\"failed\":$fail_num,\"passed\":$pass_num,\"skipped\":0}\n"
    if [ $fail_num -ne 0 ]; then
        echo '### Android模型测试失败，测试终止！'
        failed
    fi
    if [ "$OPENCL_CHANGE" ]; then
        printf "TEST_NAME_ANDROID_MODEL_OPENCL_TEST_$1: Android_$1模型测试\nTEST_CASE_AMOUNT_ANDROID_MODEL_TEST_$1: {\"blocked\":0,\"failed\":$fail_cl_num,\"passed\":$pass_cl_num,\"skipped\":0}\n"
        if [ $fail_cl_num -ne 0 ]; then
            echo '### Android OpenCL后端模型测试失败，测试终止！'
            failed
        fi
    fi
}
android_unit_test_low_memory() {
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/lowMemory 0 1 1 $1 2"
    if [ $? -ne 0 ]; then
        echo '### Android 64位Low Memory, precision=1 单元测试失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/lowMemory 0 2 1 $1 2"
    if [ $? -ne 0 ]; then
        echo '### Android 64位Low Memory, precision=2 单元测试失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/lowMemory 0 1 1 $1"
    if [ $? -ne 0 ]; then
        echo '### Android 64位 权值量化调用1x1Strassen, precision=1 单元测试失败，测试终止！'
        failed
    fi
    adb shell "cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.&&./run_test.out op/lowMemory 0 2 1 $1"
    if [ $? -ne 0 ]; then
        echo '### Android 64位 权值量化调用1x1Strassen, precision=2 单元测试失败，测试终止！'
        failed
    fi
}

android_test() {
    pushd project/android
    # 1. build Android32
    mkdir build_32
    pushd build_32
    ../build_32.sh -DMNN_BUILD_TRAIN=OFF -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DMNN_OPENCL=true
    android32_build_wrong=$[$? > 0]
    mnn32_size=$(ls -lh libMNN.so | awk '{print $5}')
    expr32_size=$(ls -lh libMNN_Express.so | awk '{print $5}')
    printf "TEST_NAME_ANDROID_32: Android32编译测试(libMNN.so - %s, libMNN_Express.so - %s)\nTEST_CASE_AMOUNT_ANDROID_32: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
           $mnn32_size $expr32_size $android32_build_wrong $[1 - $android32_build_wrong]
    if [ $android32_build_wrong -ne 0 ]; then
        echo '### Android32编译失败，测试终止！'
        failed
    fi
    ../updateTest.sh
    android_unit_test 32
    android_model_test 32
    popd

    # 3. build Android64
    mkdir build_64
    pushd build_64
    ../build_64.sh -DMNN_BUILD_TRAIN=OFF -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DMNN_ARM82=true -DMNN_OPENCL=true -DMNN_LOW_MEMORY=true
    android64_build_wrong=$[$? > 0]
    mnn64_size=$(ls -lh libMNN.so | awk '{print $5}')
    expr64_size=$(ls -lh libMNN_Express.so | awk '{print $5}')
    printf "TEST_NAME_ANDROID_64: Android64编译测试(libMNN.so - %s, libMNN_Express.so - %s)\nTEST_CASE_AMOUNT_ANDROID_64: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" \
            $mnn64_size $expr64_size $android64_build_wrong $[1 - $android64_build_wrong]
    if [ $android64_build_wrong -ne 0 ]; then
        echo '### Android64编译失败，测试终止！'
        failed
    fi

    # 4. test Android64
    ../updateTest.sh
    android_unit_test 64
    android_unit_test_low_memory 64
    android_model_test 64
    popd

    popd
}

case "$1" in
    local)
        pushd build
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        pymnn_test
        ;;
    linux)
        doc_check
        static_check
        py_check
        linux_build 1
        coverage_init
        unit_test
        model_test
        onnx_convert_test
        tf_convert_test
        tflite_convert_test
        torch_convert_test
        ptq_test
        pymnn_test
        opencv_test
        llm_test
        coverage_report
        ;;
    android)
        android_static_build
        android_test
        ;;
    *)
        $1
        echo $"Usage: $0 {local|linux|android|func}"
        exit 2
esac
exit $?
