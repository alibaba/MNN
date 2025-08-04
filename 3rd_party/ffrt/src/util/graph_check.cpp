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
#include "util/graph_check.h"

namespace ffrt {
void GraphCheckCyclic::AddVetexByLabel(uint64_t label)
{
    if (vetexes.count(label) == 0) {
        std::list<uint64_t> labelVertex;
        vetexes[label] = labelVertex;
    }
}

uint32_t GraphCheckCyclic::EdgeNum(void)
{
    uint32_t num = 0;
    for (auto &t : vetexes) {
        num += t.second.size();
    }
    return num;
}

uint32_t GraphCheckCyclic::VertexNum(void) const
{
    return vetexes.size();
}

void GraphCheckCyclic::AddEdgeByLabel(uint64_t startLabel, uint64_t endLabel)
{
    std::map<uint64_t, std::list<uint64_t>>::iterator it = vetexes.find(startLabel);
    if (it != vetexes.end()) {
        it->second.push_back(endLabel);
    }
}

void GraphCheckCyclic::RemoveEdgeByLabel(uint64_t endLabel)
{
    std::map<uint64_t, std::list<uint64_t>>::iterator it = vetexes.find(endLabel);
    if (it != vetexes.end()) {
        it->second.clear();
    }
}

bool GraphCheckCyclic::IsCyclicDfs(uint64_t v, std::map<uint64_t, struct VertexStatus>& vertexStatus)
{
    if (!vertexStatus[v].visited) {
        vertexStatus[v].visited = true;
        vertexStatus[v].recStack = true;
        std::list<uint64_t>::iterator i;
        std::map<uint64_t, std::list<uint64_t>>::iterator it = vetexes.find(v);
        for (i = it->second.begin(); i != it->second.end(); ++i) {
            if (!vertexStatus[*i].visited && IsCyclicDfs(*i, vertexStatus)) {
                return true;
            } else if (vertexStatus[*i].recStack) {
                return true;
            }
        }
    }

    vertexStatus[v].recStack = false;
    return false;
}

bool GraphCheckCyclic::IsCyclic(void)
{
    std::map<uint64_t, struct VertexStatus> vertexStatus;
    for (const auto &t : vetexes) {
        vertexStatus[t.first].visited = false;
        vertexStatus[t.first].recStack = false;
    }

    for (const auto &t : vetexes) {
        if (!vertexStatus[t.first].visited && IsCyclicDfs(t.first, vertexStatus)) {
            return true;
        }
    }

    return false;
}
}