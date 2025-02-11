// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import com.alibaba.mls.api.ApplicationUtils;
import com.alibaba.mnnllm.android.utils.FileUtils;

public interface GenerateResultProcessor {

    void process(String progress);
    String getDisplayResult();

    String getRawResult();
    void generateBegin();

    class NormalGenerateResultProcessor implements GenerateResultProcessor {
        protected StringBuilder rawStringBuilder;

        public NormalGenerateResultProcessor() {
            rawStringBuilder = new StringBuilder();
        }

        @Override
        public void process(String progress) {
            if (progress != null) {
                rawStringBuilder.append(progress);
            }
        }

        @Override
        public String getDisplayResult() {
            return rawStringBuilder.toString();
        }

        @Override
        public String getRawResult() {
            return rawStringBuilder.toString();
        }

        @Override
        public void generateBegin() {

        }
    }

    class R1GenerateResultProcessor extends NormalGenerateResultProcessor  {

        private final String thinkingPrefix;
        private long generateBeginTime;
        private boolean hasThinkProcessed;
        private final String thinkCompletePrefix;

        private StringBuilder displayStringBuilder;
        private boolean thinkStarted;

        private boolean processEnded = false;

        public R1GenerateResultProcessor(String thinkingPrefix, String thinkCompletePrefix) {
            displayStringBuilder = new StringBuilder();
            displayStringBuilder.append(thinkingPrefix).append("\n");
            this.thinkCompletePrefix = thinkCompletePrefix;
            this.thinkingPrefix = thinkingPrefix;
        }

        @Override
        public String getRawResult() {
            return processEnded ? rawStringBuilder.toString().trim() + "<|end_of_sentence|>" : super.getRawResult();
        }

        @Override
        public void generateBegin() {
            super.generateBegin();
            this.generateBeginTime = System.currentTimeMillis();
        }

        @Override
        public String getDisplayResult() {
            return displayStringBuilder.toString();
        }

        @Override
        public void process(String progress) {
            if (progress == null) {
                processEnded = true;
                return;
            }
            if (hasThinkProcessed) {
                rawStringBuilder.append(progress);
            }
            rawStringBuilder.append(progress);
            //uncomment this to debug long output
            //FileUtils.saveStringToFile(ApplicationUtils.get(), "r1_result.txt", rawStringBuilder.toString());
            if (progress.contains("</think>")) {
                progress = progress.replace("</think>", "\n");
                long thinkTime = (System.currentTimeMillis() - this.generateBeginTime) / 1000;
                displayStringBuilder.replace(0, thinkingPrefix.length(),
                        thinkCompletePrefix.replace("ss", String.valueOf(thinkTime)));
                hasThinkProcessed = true;
            } else if (!hasThinkProcessed) {
                if (!thinkStarted) {
                    displayStringBuilder.append("> ");
                    thinkStarted = true;
                }
                if (progress.contains("\n")) {
                    progress = progress.replace("\n", "\n> ");
                }
            }
            displayStringBuilder.append(progress);
        }
    }

}
