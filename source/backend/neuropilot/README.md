# 导出并量化模型

## 编译 MNN

```plaintext
mkdir build
cmake ../ -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_LLM=ON -DMNN_NEUROPILOT=ON
make -j4
```

准备测试文件 512.txt
```plaintext
我在写一个以魔法世界为主题的中篇小说。我已经完成了开头，请你在我完成的部分的基础上续写该小说。我已经完成的部分如下：<<<《奥术回廊的第七根蜡烛》第一章：褪色的录取函1. 灰烬中的秘密伊莱·维瑟兰用拇指摩挲着信封边缘的金色火漆印，那枚印记在他触碰的瞬间微微发烫，仿佛活物般收缩了一下。他犹豫片刻，还是撕开了信封。羊皮纸在展开的刹那化作细碎的灰烬，像一群受惊的飞蛾般四散飘落，只留下几行暗红色的字迹浮现在空中：“以三滴晨露与午夜叹息为凭，汝已被奥术回廊学院录取。”“报到日：下弦月之夜。”“携带物品：一根未染血的银针、一颗自愿献出的牙齿、一段不为人知的记忆。”伊莱皱眉，指尖轻轻触碰最后一行几乎被烧毁的小字——那里本该是校训的位置，却只剩下半句残缺的警告：“知识是……”后面的字迹被某种力量抹去，只留下一道焦黑的裂痕，像是被火焰舔舐过的皮肤。“知识是蜜糖还是毒药？”母亲的声音突然从身后传来，枯瘦的手指轻轻搭上他的肩膀。伊莱猛地回头，发现她不知何时站在了阴影里，烛光只照亮她半边脸，另一半隐没在黑暗中，像是被什么东西啃噬过。“你父亲收到录取函时，最后一行是完整的。”她低声说，指尖滑向伊莱后颈的胎记——那团火焰形状的印记此刻正隐隐发烫，就像七岁那年他无意间触碰祖父的魔法书时一样。“别让他们发现你能看见‘不该看的东西’。”她最后叮嘱道，声音轻得像一阵风，随后转身消失在走廊尽头，只留下一缕若有若无的草药苦味。2. 雨中学院报到日当天，暴雨倾盆。奥术回廊学院矗立在悬崖边缘，哥特式的尖顶刺破铅灰色的天空，黑曜石外墙不断渗出粘稠的黑色液体，像是一头受伤的巨兽在流血。新生们踩着骨白色的台阶向上攀登。...>>>
```

## 导出
有两种方案
### 使用 smoothquant
增加 `--smooth --act_bit=16 --quant_block=0 --lm_quant_bit=16 --quant_bit=4 --seperate_embed --sym` 以导出 mnn
        
eg: 
```
python3 llmexport.py --path /Users/xtjiang/.cache/modelscope/hub/models/Qwen/Qwen3-4B --export mnn --smooth --act_bit=16 --quant_block=0 --lm_quant_bit=16 --seperate_embed --quant_bit=4 --sym
```
            
### 两步量化（相比前者快很多，但目前效果较差）
        
- 使用 `--quant_block=0 --lm_quant_bit=16 --seperate_embed --quant_bit=4 --sym` 导出 mnn 
- 执行 `./quantize_llm ../transformers/llm/export/model/config.json 512.txt 16 temp.bin && cp temp.bin ../transformers/llm/export/model/llm.mnn` 量化特征
            

# 使用 `compilefornpu` 生成 tflite

## 使用 generateLlmIO 生成 input / output

```
./generateLlmIO ../transformers/llm/export/model  ../transformers/llm/export/model/testdir
```

## 编辑如下的json文件：npu.json

```json
{
    "name":"MLDA",
    "skips":[
        "/Reshape_output_0",
        "/Gather_3_output_0",
        "/Gather_4_output_0"
    ],
    "testdir":[
        "testdir/1",
        "testdir/128"
    ]
}

```

将 testdir 中的路径修改为 model/testdir 对应路径

##  执行 `compilefornpu`

rm -r res
mkdir res
./seperatenpu ../transformers/llm/export/model/llm.mnn res/temp.bin npu.json

当前目录下会增加 `npu_postreat.json`

# 使用 `npu_convert.py` 将 tflite 编译为 dla

## 下载 sdk 并配置环境变量

下载 NEURON_SDK ，并修改 `~/.bashrc` ，增加对应的路径
eg: 
```
export NEURON_SDK=/home/xiaying/third/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk

```

## 执行转换脚本

```
python3 ../source/backend/neuropilot/npu_convert.py npu_postreat.json > 1
```

## 【可选】清除中间产物

```
rm res/*.tflite
rm res/*.dla
```

# model 目录构建

在 build 目录下执行
```
rm -r model
mv ../transformer/llm/export/model model
```

## npu 相关文件复制

将 res/ 下的构建产物复制到 model 目录下，并用 res/temp.bin 重命名为 model/llm_npu.mnn 

```plaintext
rm -r model/res
mv res model/res
mv model/res/temp.bin model/llm_npu.mnn
```

## 增加 `model/config_npu.json`

```json
{
    "llm_model": "llm_npu.mnn",
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "chunk_limits":[128, 1],
    "memory": "low",
    "sampler_type": "penalty",
    "penalty": 1.1
}
```

## 将 model/llm.mnn.weight 删除（可选）

*   llm.mnn 和 llm.mnn.weight 不再需要，如果不需要对比 CPU / GPU 的性能可以将它们删除
    

```plaintext
rm model/llm.mnn
rm model/llm.mnn.weight
```

# 测试

## 测试资源准备

*   将构建产物传到设备上：
    

```plaintext
#!/bin/bash
adb shell mkdir /data/local/tmp/MNN
adb shell rm -r /data/local/tmp/MNN/model
adb push model /data/local/tmp/MNN/model
```

*   上传测试文件 512.txt


```
adb push 512.txt /data/local/tmp/MNN/512.txt
```

## 运行测试程序

*   编译并上传

在 MNN 根目录下逐句操作：

```plaintext
cd project/android/
mkdir build_64
cd build_64
../build_64.sh -DMNN_NEUROPILOT=ON -DMNN_WITH_PLUGIN=ON -DMNN_BUILD_LLM=ON
../updateTest.sh
```

*   运行程序

```plaintext
cd project/android/build_64
../testCommon.sh ./llm_demo model/config_npu.json 512.txt
```
