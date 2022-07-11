#!/bin/bash

usage() {
    echo "Usage: $0 [--sync-adb] [--pc] [--android] [--ios] [--mnn] [--torch] [--tf]"
    echo -e "\t--sync-adb sync test models and benchmark tool"
    echo -e "\t--pc test for pc"
    echo -e "\t--android test for android"
    echo -e "\t--ios test for ios"
    echo -e "\t--mnn test"
    echo -e "\t--torch test"
    echo -e "\t--tf test"
    exit 1
}

sync_adb=false
test_for_pc=false
test_for_android=false
test_for_ios=false
test_mnn=false
test_torch=false
test_tf=false
while getopts ":h-:" opt; do
  case "$opt" in
    -)
        case "$OPTARG" in
            sync-adb) sync_adb=true ;;
            pc) test_for_pc=true ;;
            android) test_for_android=true ;;
            ios) test_for_ios=true ;;
            mnn) test_mnn=true ;;
            torch) test_torch=true ;;
            tf) test_tf=true ;;
            *) usage ;;
        esac ;;
    h|? ) usage ;;
  esac
done

if ! command -v jq &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        apt install -y jq
    fi
fi

rm -rf result && mkdir result

MNN_HOME=/data/local/tmp/MNN
TORCH_HOME=/data/local/tmp/torch
TFLITE_HOME=/data/local/tmp/tflite
if $test_for_android; then
    if ! command -v adb &> /dev/null; then
        echo 'adb not found'
        exit 1
    fi
    if $sync_adb; then
        adb push dist/android/* /data/local/tmp
        adb shell "mkdir -p $MNN_HOME/models"
        adb push models/mnn/*.mnn $MNN_HOME/models
        adb shell "mkdir -p $TORCH_HOME/models"
        adb push models/torch_lite/*.ptl $TORCH_HOME/models
        adb shell "mkdir -p $TFLITE_HOME/models/fp16"
        adb push models/tflite/*.tflite $TFLITE_HOME/models
        adb push models/tflite/fp16/*.tflite $TFLITE_HOME/models/fp16
    fi
fi
if $test_for_ios; then
    gem install bundler xcodeproj
fi

bench_common() {
    echo "$1 begin"
    for i in $(seq $(cat "models/config.json" | jq "length")); do
        model_meta=$(cat "models/config.json" | jq ".[$i-1]")
        model_name=$(echo $model_meta | jq -r ".model")
        local input_layers=() input_shapes=() input_dtypes=() bench_args
        for j in $(seq $(echo $model_meta | jq ".input_layers|length")); do
            input_layers+=($(echo $model_meta | jq -r ".input_layers[$j-1]"))
            input_shapes+=($(echo $model_meta | jq -r ".input_shapes[$j-1]"))
            input_dtypes+=($(echo $model_meta | jq -r ".input_dtypes[$j-1]"))
        done
        $1 "$model_name" "$bench_args"
    done
    echo "$1 end"
}

# mnn android
if $test_mnn; then
    mnn_android_arm32_bench() {
        adb shell "export LD_LIBRARY_PATH=$MNN_HOME/arm32 && $MNN_HOME/arm32/benchmark.out $MNN_HOME/models 10 5 0 1 2>&1" >> result/mnn_android_arm32.txt
    }
    mnn_android_arm64_bench() {
        adb shell "export LD_LIBRARY_PATH=$MNN_HOME/arm64 && $MNN_HOME/arm64/benchmark.out $MNN_HOME/models 10 5 0 1 2>&1" >> result/mnn_android_arm64.txt
    }
    mnn_android_armv82_bench() {
        adb shell "export LD_LIBRARY_PATH=$MNN_HOME/arm64 && $MNN_HOME/arm64/benchmark.out $MNN_HOME/models 10 5 0 1 2 2>&1" >> result/mnn_android_fp16.txt
    }
    mnn_android_opencl_bench() {
        adb shell "export LD_LIBRARY_PATH=$MNN_HOME/arm64 && $MNN_HOME/arm64/benchmark.out $MNN_HOME/models 10 5 3 1 2>&1" >> result/mnn_android_opencl.txt
    }
    if $test_for_pc; then
        ./MNN/build/benchmark.out models/mnn 10 5 0 1 2>&1 >> result/mnn_pc_cpu.txt # CPU
        #./MNN/build/benchmark.out models/mnn 10 5 2 2&>1 >> result/mnn_pc_cuda.txt # CUDA
        #python bench_pc.py -f mnn --modeldir models --thread-num 1 --backend cpu
        #python bench_pc.py -f mnn --modeldir models --thread-num 1 --backend cuda
    fi
    if $test_for_android; then
        mnn_android_arm32_bench
        mnn_android_arm64_bench
        mnn_android_armv82_bench
        mnn_android_opencl_bench
    fi
fi

# pytorch mobile android
if $test_torch; then
    torch_compatible() {
        input_shapes_str=$(IFS=";" ; echo "${input_shapes[*]}")
        input_dtypes_str=$(IFS=";" ; echo "${input_dtypes[*]}")
        input_dtypes_str=$(python -c "print('$input_dtypes_str'.replace('int', 'int64'))")
        input_memory_format_str=$(python -c "print(';'.join(['contiguous_format'] * ${#input_shapes[@]}))")
        bench_args="--input_dims='$input_shapes_str' --input_type='$input_dtypes_str' --input_memory_format='$input_memory_format_str'"
    }
    torch_android_arm32() {
        torch_compatible
        adb shell "$TORCH_HOME/arm32/speed_benchmark_torch --model=$TORCH_HOME/models/$1.ptl $bench_args 2>&1" >> result/torch_android_arm32.txt
    }
    torch_android_arm64() {
        torch_compatible
        adb shell "$TORCH_HOME/arm64/speed_benchmark_torch --model=$TORCH_HOME/models/$1.ptl $bench_args 2>&1" >> result/torch_android_arm64.txt
    }
    if $test_for_pc; then
        python -m pip uninstall -y torch # May be custom torch wheel (build for metal optimize), uninstall it
        python -m pip install torch==1.11.0
        python bench_pc.py -f torch --modeldir models --thread-num 1 --backend cpu
        python bench_pc.py -f torch --modeldir models --thread-num 1 --backend cuda
    fi
    # torch android arm32
    if $test_for_android; then
        bench_common torch_android_arm32
        bench_common torch_android_arm64
    fi

    # iOS app generated by ios/TestApp/benchmark/setup.rb be failed even a empty test case, say: Unknown custom class type quantized.Conv2dPackedParamsBase
    #if $test_for_ios; then
    #    pushd pytorch
    #    rm -rf build_ios && cp -r build_ios_arm64 build_ios # ios/TestApp/benchmark/setup.rb hardcode build_ios
    #    cp -r ../models/torch_lite ios/TestApp/models
    #    pushd ios/TestApp/benchmark && ruby setup.rb -lite && popd
    #    popd
    #fi
fi

# tflite android
if $test_tf; then
    tflite_compatible() {
        local i j input_num=${#input_layers[@]}
        local origin_input_layers=("${input_layers[@]}") origin_input_shapes=("${input_shapes[@]}")
        local model_meta=$(cat models/tflite/config.json | jq "map(select(.model == \"$model_name\"))|.[0]")
        input_shapes=()
        input_layers=()
        for i in $(seq $input_num); do
            for j in $(seq $input_num); do
                if [[ $(echo $model_meta | jq -r ".inputs[$i-1]") == ${origin_input_layers[$j-1]} ]]; then
                    input_shapes+=(${origin_input_shapes[$j-1]})
                    break
                fi
            done
            input_layers+=($(echo $model_meta | jq -r ".inner_inputs[$i-1]"))
        done
        input_layers_str=$(IFS="," ; echo "${input_layers[*]}")
        input_shapes_str=$(IFS=":" ; echo "${input_shapes[*]}")
        bench_args="--input_layer='$input_layers_str' --input_layer_shape='$input_shapes_str' --warmup_runs=1 --num_runs=20"
    }
    tflite_android_arm32() {
        tflite_compatible
        adb shell "$TFLITE_HOME/arm32/benchmark_model_plus_flex --graph=$TFLITE_HOME/models/$1.tflite $bench_args --num_threads=1 2>&1" >> result/tflite_android_arm32.txt
    }
    tflite_android_arm64() {
        tflite_compatible
        adb shell "$TFLITE_HOME/arm64/benchmark_model_plus_flex --graph=$TFLITE_HOME/models/$1.tflite $bench_args --num_threads=1 2>&1" >> result/tflite_android_arm64.txt
    }
    tflite_android_fp16() {
        tflite_compatible
        adb shell "$TFLITE_HOME/arm64/benchmark_model_plus_flex --graph=$TFLITE_HOME/models/fp16/$1.tflite $bench_args --num_threads=1 2>&1" >> result/tflite_android_fp16.txt
    }
    tflite_android_gpu() {
        tflite_compatible
        adb shell "$TFLITE_HOME/arm64/benchmark_model_plus_flex --use_gpu=true --graph=$TFLITE_HOME/models/$1.tflite $bench_args --num_threads=1 2>&1" >> result/tflite_android_gpu.txt
    }
    if $test_for_pc; then
        python -m pip install tensorflow==2.7.0
        python bench_pc.py -f tf --modeldir models --thread-num 1 --backend cpu
        python bench_pc.py -f tf --modeldir models --thread-num 1 --backend cuda
    fi
    if $test_for_android; then
        bench_common tflite_android_arm32
        bench_common tflite_android_arm64
        bench_common tflite_android_fp16
        bench_common tflite_android_gpu
    fi
    if $test_for_ios; then
        tflite_gen_params() {
            tflite_compatible
            echo \
"""{
    \"benchmark_name\": \"${model_name}_benchmark\",
    \"num_threads\" : \"1\",
    \"num_runs\" : \"20\",
    \"warmup_runs\" : \"1\",
    \"graph\" : \"${model_name}.tflite\",
    \"input_layer\" : \"${input_layers}\",
    \"input_layer_shape\" : \"${input_shapes}\",
    \"run_delay\" : \"-1\"
}""" > models/tflite/${model_name}_benchmark_params.json
        }
        bench_common tflite_gen_params
    fi
fi
