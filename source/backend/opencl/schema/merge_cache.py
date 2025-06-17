import flatbuffers
from CLCache import Cache, BackendInfo, Autotuning, GemmInfo

def load_backend_infos(file_path):
    with open(file_path, 'rb') as f:
        buf = bytearray(f.read())
    cache = Cache.Cache.GetRootAs(buf, 0)
    backends = []
    for i in range(cache.BackendsLength()):
        backend = cache.Backends(i)
        backends.append(backend)
    return backends

def load_tune_infos(backends):
    original_map = {}
    for backend in backends:
        device_name = backend.DeviceName()
        tunings = {}
        for i in range(backend.TuningsLength()):
            tune = backend.Tunings(i)
            key = tune.Key()
            program = tune.Name()
            md5 = tune.Md5()
            global_size = [tune.GloablSize(j) for j in range(tune.GloablSizeLength())]
            local_size = [tune.LocalSize(j) for j in range(tune.LocalSizeLength())]
            cost_time = tune.TimeCost()
            tunings[(key, tuple(global_size))] = (local_size, cost_time, program, md5)

        #gemm tune info
        for i in range(backend.GemmLength()):
            tune = backend.Gemm(i)
            key = 'Xgemm_tune'
            md5 = tune.Md5()
            gemm_size = [tune.GemmSize(j) for j in range(tune.GemmSizeLength())]
            param_info = [tune.ParamInfo(j) for j in range(tune.ParamInfoLength())]
            tunings[(key, tuple(gemm_size))] = (param_info, 0, "matmul_params_buf", md5)
        original_map[device_name] = tunings
    return original_map

def create_backend_info(new_backends, original_backends):

    original_map = load_tune_infos(original_backends)
    new_map = load_tune_infos(new_backends)

    for ver_dev in new_map:
        if ver_dev in original_map:
            new_tune = new_map[ver_dev]
            original_tune = original_map[ver_dev]
            for key in new_tune:
                if key not in original_tune:
                    original_tune[key] = new_tune[key]
        else:
            original_map[ver_dev] = new_map[ver_dev]
    
    return original_map

def build_cache(nested_dict):
    """将嵌套字典转换为 FlatBuffers 的 Cache 结构"""
    builder = flatbuffers.Builder()

    # ====================== 构建 BackendInfo 列表 ======================
    backend_offsets = []
    for device_name, autotune_dict in nested_dict.items():
        # 构建字符串
        device_name_offset = builder.CreateString(device_name)

        # 构建 Autotuning 条目
        tuning_offsets = []
        gemm_offsets = []
        for (key, global_size), (local_size, time_cost, name, md5) in autotune_dict.items():
            #print(name)
            if key == 'Xgemm_tune':
                # 构建 GemmSize 向量 (倒序填充)
                GemmInfo.GemmInfoStartGemmSizeVector(builder, len(global_size))
                for n in reversed(global_size):
                    builder.PrependUint32(n)
                global_size_offset = builder.EndVector()
            
                # 构建 ParamInfo 向量 (倒序填充)
                GemmInfo.GemmInfoStartParamInfoVector(builder, len(local_size))
                for n in reversed(local_size):
                    builder.PrependUint32(n)
                local_size_offset = builder.EndVector()

                # 构建 md5 字符串
                md5_offset = builder.CreateString(md5)
            
                # 构建 Autotuning 对象
                GemmInfo.GemmInfoStart(builder)
                GemmInfo.GemmInfoAddGemmSize(builder, global_size_offset)
                GemmInfo.GemmInfoAddParamInfo(builder, local_size_offset)
                GemmInfo.GemmInfoAddMd5(builder, md5_offset)
                gemm_offsets.append(GemmInfo.GemmInfoEnd(builder))
            else:
                # 构建字符串
                key_offset = builder.CreateString(key)
            
                # 构建 globalSize 向量 (倒序填充)
                Autotuning.AutotuningStartGloablSizeVector(builder, len(global_size))
                for n in reversed(global_size):
                    builder.PrependUint32(n)
                global_size_offset = builder.EndVector()
            
                # 构建 localSize 向量 (倒序填充)
                Autotuning.AutotuningStartLocalSizeVector(builder, len(local_size))
                for n in reversed(local_size):
                    builder.PrependUint32(n)
                local_size_offset = builder.EndVector()

                # 构建name字符串
                name_offset = builder.CreateString(name)

                # 构建md5字符串
                md5_offset = builder.CreateString(md5)
            
                # 构建 Autotuning 对象
                Autotuning.AutotuningStart(builder)
                Autotuning.AutotuningAddKey(builder, key_offset)
                Autotuning.AutotuningAddGloablSize(builder, global_size_offset)
                Autotuning.AutotuningAddLocalSize(builder, local_size_offset)
                Autotuning.AutotuningAddTimeCost(builder, time_cost)
                Autotuning.AutotuningAddName(builder, name_offset)
                Autotuning.AutotuningAddMd5(builder, md5_offset)
                tuning_offsets.append(Autotuning.AutotuningEnd(builder))

        # 构建 tunings 向量
        BackendInfo.BackendInfoStartTuningsVector(builder, len(tuning_offsets))
        for offset in reversed(tuning_offsets):
            builder.PrependUOffsetTRelative(offset)
        tunings_offset = builder.EndVector()

        # 构建 gemm 向量
        BackendInfo.BackendInfoStartGemmVector(builder, len(gemm_offsets))
        for offset in reversed(gemm_offsets):
            builder.PrependUOffsetTRelative(offset)
        gemm_offsets = builder.EndVector()

        # 构建 BackendInfo
        BackendInfo.BackendInfoStart(builder)
        BackendInfo.BackendInfoAddDeviceName(builder, device_name_offset)
        BackendInfo.BackendInfoAddTunings(builder, tunings_offset)
        BackendInfo.BackendInfoAddGemm(builder, gemm_offsets)
        backend_offsets.append(BackendInfo.BackendInfoEnd(builder))

    # ====================== 构建最终 Cache ======================
    # 构建 backends 向量
    Cache.CacheStartBackendsVector(builder, len(backend_offsets))
    for offset in reversed(backend_offsets):
        builder.PrependUOffsetTRelative(offset)
    backends_offset = builder.EndVector()

    # 构建根对象
    Cache.CacheStart(builder)
    Cache.CacheAddBackends(builder, backends_offset)
    cache = Cache.CacheEnd(builder)

    builder.Finish(cache)
    return builder.Output()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python merge_cache.py <primary_file> <total_file> <output_file>")
        print("Example: python merge_cache.py mnn_cachefile.bin mnn_cachefile_total.bin new_cache.bin")
        sys.exit(1)
    original_backends = load_backend_infos(sys.argv[1])
    new_backends = load_backend_infos(sys.argv[2])
    original_map = create_backend_info(new_backends, original_backends)
    #print(original_map)
    binary_data = build_cache(original_map)
    with open(sys.argv[3], "wb") as f:
        f.write(binary_data)