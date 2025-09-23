/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
#ifndef __GRAPH_CHECK_H__
#define __GRAPH_CHECK_H__
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <list>

namespace ffrt {
struct VertexStatus {
    bool visited;
    bool recStack;
};

class GraphCheckCyclic {
    std::map<uint64_t, std::list<uint64_t>> vetexes;
    bool IsCyclicDfs(uint64_t v, std::map<uint64_t, struct VertexStatus>& vertexStatus);
public:
    void AddVetexByLabel(uint64_t label);
    void AddEdgeByLabel(uint64_t startLabel, uint64_t endLabel);
    void RemoveEdgeByLabel(uint64_t endLabel);
    uint32_t EdgeNum(void);
    uint32_t VertexNum(void) const;
    bool IsCyclic();
};
}
#endif