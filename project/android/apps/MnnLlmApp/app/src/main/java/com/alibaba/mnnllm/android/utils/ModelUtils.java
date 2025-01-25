// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.annotation.SuppressLint;

import com.alibaba.mls.api.HfRepoItem;
import com.alibaba.mnnllm.android.R;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ModelUtils {

    public static int getDrawableId(String modelName) {
        if (modelName == null) {
            return 0;
        }
        String modelLower = modelName.toLowerCase();
        if (modelLower.contains("qwen") || modelLower.contains("qwq")) {
            return R.drawable.qwen_icon;
        } else if (modelLower.contains("llama") || modelLower.contains("mobilellm")) {
            return R.drawable.llama_icon;
        } else if (modelLower.contains("smo")) {
            return R.drawable.smolm_icon;
        } else if (modelLower.contains("phi")) {
            return R.drawable.phi_icon;
        } else if (modelLower.contains("baichuan")) {
            return R.drawable.baichuan_icon;
        } else if (modelLower.contains("yi")) {
            return R.drawable.yi_icon;
        } else if (modelLower.contains("glm") || modelLower.contains("codegeex")) {
            return R.drawable.chatglm_icon;
        } else if (modelLower.contains("reader")) {
            return R.drawable.jina_icon;
        } else if (modelLower.contains("deepseek")) {
            return R.drawable.deepseek_icon;
        } else if (modelLower.contains("internlm")) {
            return R.drawable.internlm_icon;
        } else if (modelLower.contains("gemma")) {
            return R.drawable.gemma_icon;
        }
        return 0;
    }

    @SuppressLint("DefaultLocale")
    public static String generateBenchMarkString(HashMap<String, Object> metrics) {
        if (metrics.containsKey("total_timeus")) {
            return generateDiffusionBenchMarkString(metrics);
        }
        long promptLen = (long) metrics.get("prompt_len");
        long decodeLen = (long) metrics.get("decode_len");
        long prefillTimeUs = (long) metrics.get("prefill_time");
        long decodeTimeUs = (long) metrics.get("decode_time");
        // Calculate speeds in tokens per second
        double promptSpeed = (prefillTimeUs > 0) ? (promptLen / (prefillTimeUs / 1_000_000.0)) : 0.0;
        double decodeSpeed = (decodeTimeUs > 0) ? (decodeLen / (decodeTimeUs / 1_000_000.0)) : 0.0;
        return String .format("Prefill: %d tokens, %.2f tokens/s\nDecode: %d tokens, %.2f tokens/s",
                promptLen,promptSpeed, decodeLen,decodeSpeed);
    }

    @SuppressLint("DefaultLocale")
    public static String generateDiffusionBenchMarkString(HashMap<String, Object> metrics) {
        double totalDuration = (long) metrics.get("total_timeus") * 1.0 / 1_000_000.0;
        return String.format("Generate time: %.2f s", totalDuration);
    }

    private static final Set<String> blackList = new HashSet<>();
    static {
        blackList.add("taobao-mnn/bge-large-zh-MNN");//embedding
        blackList.add("taobao-mnn/gte_sentence-embedding_multilingual-base-MNN");//embedding
        blackList.add("taobao-mnn/QwQ-32B-Preview-MNN");//too big
        blackList.add("taobao-mnn/codegeex2-6b-MNN");//not for chat
        blackList.add("taobao-mnn/chatglm-6b-MNN");//deprecated
        blackList.add("taobao-mnn/chatglm2-6b-MNN");
        blackList.add("taobao-mnn/stable-diffusion-v1-5-mnn-general");//in android, we use opencl version
    }

    //list that are more stable
    private static final Set<String> goodList = new HashSet<>();
    static {
        goodList.add("taobao-mnn/Qwen2.5-0.5B-Instruct-MNN");
        goodList.add("taobao-mnn/Qwen2.5-1.5B-Instruct-MNN");
        goodList.add("taobao-mnn/Qwen2.5-7B-Instruct-MNN");
        goodList.add("taobao-mnn/Qwen2.5-3B-Instruct-MNN");
        goodList.add("taobao-mnn/gemma-2-2b-it-MNN");
    }

    private static boolean isBlackListPattern(String modelName) {
        return modelName.contains("qwen1.5") || modelName.contains("qwen-1");
    }
    public static List<HfRepoItem> processList(List<HfRepoItem> hfRepoItems) {
        List<HfRepoItem> goodItems = new ArrayList<>();
        List<HfRepoItem> chatItems = new ArrayList<>();
        List<HfRepoItem> otherItems = new ArrayList<>();
        for (HfRepoItem item : hfRepoItems) {
            String modelIdLowerCase = item.getModelId().toLowerCase();
            if (blackList.contains(item.getModelId()) || isBlackListPattern(modelIdLowerCase)) {
                continue;
            }
            if (goodList.contains(item.getModelId())) {
                goodItems.add(item);
            } else if (modelIdLowerCase.contains("chat")) {//optimized for chat, should at top
                chatItems.add(item);
            } else {
                otherItems.add(item);
            }
        }
        List<HfRepoItem> result = new ArrayList<>(chatItems.size() + otherItems.size());
        result.addAll(goodItems);
        result.addAll(chatItems);
        result.addAll(otherItems);
        return result;
    }

    public static boolean isAudioModel(String modelName) {
        return modelName.toLowerCase().contains("audio");
    }

    public static boolean isDiffusionModel(String modelName) {
        return modelName.toLowerCase().contains("stable-diffusion");
    }

    public static ArrayList<String> generateSimpleTags(String modelName) {
        String[] splits = modelName.split("-");
        ArrayList<String> tags = new ArrayList<>();
        boolean isDiffusion = ModelUtils.isDiffusionModel(modelName);
        if (splits.length > 1 && !isDiffusion) {
            String brand = splits[0];
            tags.add(brand.toLowerCase());
        }
        for (int i = 1; i < splits.length; i++) {
            String tag = splits[i];
            if (tag.toLowerCase().matches("^[\\\\.0-9]+[mb]$")) {
                tags.add(tag.toLowerCase());
            }
        }
        if (isDiffusion) {
            tags.add("diffusion");
        } else {
            tags.add("text");
            if (ModelUtils.isAudioModel(modelName)) {
                tags.add("audio");
            } else if (ModelUtils.isVisualModel(modelName)) {
                tags.add("visual");
            }
        }
        return tags;
    }

    public static boolean isVisualModel(String modelName) {
        return modelName.toLowerCase().contains("vl");
    }
}
