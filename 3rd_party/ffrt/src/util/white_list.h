/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FFRT_WHITE_LIST
#define FFRT_WHITE_LIST
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "internal_inc/osal.h"
#include "util/ffrt_facade.h"
namespace {
constexpr char CONF_FILEPATH[] = "/etc/ffrt/ffrt_whitelist.conf";
constexpr int INDENT_SPACE_NUM = 4;
}

class WhiteList {
public:
    static const WhiteList& GetInstance()
    {
        static WhiteList instance;
        return instance;
    }

    bool IsEnabled(const std::string& functionName, bool defaultWhenAbnormal) const
    {
        auto it = whiteList_.find(functionName);
        if (it != whiteList_.end()) {
            return it->second;
        }
        // 若白名单加载失败或不在白名单中的默认返回值
        return defaultWhenAbnormal;
    }

private:
    WhiteList()
    {
        LoadFromFile();
    }

    void LoadFromFile()
    {
#ifdef OHOS_STANDARD_SYSTEM
        std::string processName = std::string(ffrt::GetCurrentProcessName());
        if (strlen(processName.c_str()) == 0) {
            FFRT_LOGW("Get process name failed.");
            return;
        }

        std::string whiteProcess;
        std::ifstream file(CONF_FILEPATH);
        std::string functionName;
        if (file.is_open()) {
            while (std::getline(file, whiteProcess)) {
                size_t pos = whiteProcess.find("{");
                if (pos != std::string::npos) {
                    functionName = whiteProcess.substr(0, pos - 1);
                    whiteList_[functionName] = false;
                } else if ((whiteProcess != "}" && whiteProcess != "") &&
                    processName.find(whiteProcess.substr(INDENT_SPACE_NUM)) != std::string::npos) {
                    whiteList_[functionName] = true;
                }
            }
        } else {
            // 当文件不存在或者无权限时默认都关
            FFRT_LOGW("white_list.conf does not exist or file permission denied");
        }
#else
        whiteList_["IsInSFFRTList"] = false;
        if (std::string(ffrt::GetCurrentProcessName()).find("CameraDaemon") != std::string::npos) {
            whiteList_["SetThreadAttr"] = true;
            whiteList_["CreateCPUWorker"] = true;
            whiteList_["HandleTaskNotifyConservative"] = true;
        }
#endif // OHOS_STANDARD_SYSTEM
    }

    WhiteList(const WhiteList&) = delete;
    WhiteList& operator=(const WhiteList&) = delete;

    std::unordered_map<std::string, bool> whiteList_;
};
#endif