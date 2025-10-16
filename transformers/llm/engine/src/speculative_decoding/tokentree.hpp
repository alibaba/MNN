//
//  tokentree.hpp
//
//  Created by MNN on 2025/09/16.
//

#ifndef TOKENTREE_HPP
#define TOKENTREE_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <queue>
#include <algorithm>
#include <sstream>
#include <map>

namespace MNN {
namespace Transformer {

class TokenTreeNode;
using NodePtr = std::shared_ptr<TokenTreeNode>;

class TokenTreeNode : public std::enable_shared_from_this<TokenTreeNode> {
public:
    int mTokenId, mNodeId, mDepth;
    double mLogProb, mCumulativeLogProb;
    std::weak_ptr<TokenTreeNode> mParent;
    std::vector<NodePtr> mChildren;

    TokenTreeNode(int tokenId, double logProb, NodePtr parent, int nodeId)
        : mTokenId(tokenId), mLogProb(logProb), mNodeId(nodeId), mParent(parent) {
        if (parent) {
            mDepth = parent->mDepth + 1;
            mCumulativeLogProb = parent->mCumulativeLogProb + logProb;
        } else {
            mDepth = 0;
            mCumulativeLogProb = logProb;
        }
    }

    void addChild(NodePtr child) {
        this->mChildren.push_back(child);
    }
};


struct TreeOutputs {
    std::vector<int> draftTokens;
    std::vector<int> positionIds;
    std::vector<std::vector<bool>> attentionMask;
    std::vector<std::vector<int>> retrieveIndices;
};

class TokenTree {
public:
    TokenTree(int topK, const int* d2tPtr = nullptr) : mTopK(topK), mD2tPtr(d2tPtr), mCounter(0) {
        mRoot = std::make_shared<TokenTreeNode>(-1, 0.0, nullptr, -1);
        mRoot->mDepth = -1;
        // init mask
        mMask.assign(topK, std::vector<bool>(topK, false));
        for (int i = 0; i < topK; i++) {
            mMask[i][i] = true;
        }
    }

    void init(const int* indices, const float* scores) {
        for (size_t i = 0; i < mTopK; i++) {
            auto node = std::make_shared<TokenTreeNode>(
                d2t(indices[i]), scores[i], mRoot, mCounter++
            );
            mRoot->addChild(node);
            mActives.push_back(node);
        }
    }

    void grow(const int* indices, const float* scores) {
        std::vector<NodePtr> candidates;
        std::map<NodePtr, int> parant2index;
        // 1. Generate all possible child nodes for each active leaf.
        for (size_t i = 0; i < mActives.size(); ++i) {
            auto parent = mActives[i];
            parant2index[parent] = i;
            for (size_t j = 0; j < mTopK; j++) {
                auto child_node = std::make_shared<TokenTreeNode>(
                    d2t(indices[i * mTopK + j]),
                    scores[i * mTopK + j],
                    parent, mCounter++
                );
                parent->addChild(child_node);
                candidates.push_back(child_node);
            }
        }

        // 2. Prune: Sort all new candidates by their cumulative log probability.
        std::sort(candidates.begin(), candidates.end(), [](const NodePtr& a, const NodePtr& b) {
            return a->mCumulativeLogProb > b->mCumulativeLogProb;
        });

        // 3. Select the top_k best nodes as the next round's active.
        std::vector<NodePtr> newActives;
        std::vector<int> parentIndices;
        for (size_t i = 0; i < std::min((size_t)mTopK, candidates.size()); ++i) {
            newActives.push_back(candidates[i]);
            parentIndices.push_back(parant2index[candidates[i]->mParent.lock()]);
        }

        // 4. update attention_mask
        std::vector<std::vector<bool>> newMask;
        size_t oldMaskSize = mMask[0].size();
        size_t newMaskSize = oldMaskSize + mTopK;
        for (size_t i = 0; i < parentIndices.size(); i++) {
            int idx = parentIndices[i];
            const auto& oldMaskLine = mMask[idx];
            std::vector<bool> newMaskLine(newMaskSize);
            std::copy(oldMaskLine.begin(), oldMaskLine.end(), newMaskLine.begin());
            std::fill(newMaskLine.begin() + oldMaskSize, newMaskLine.end(), false);
            newMaskLine[oldMaskSize + i] = true;
            newMask.emplace_back(std::move(newMaskLine));
        }
        mMask = std::move(newMask);
#if 0
        // dump
        for (int i = 0; i < mTopK; i++) {
            for (int j = 0; j < newMaskSize; j++) {
                std::cout << mMask[i][j] << ", ";
            }
            std::cout << std::endl;
        }
#endif
        // 5. Update the active leaves list.
        mActives = std::move(newActives);
    }

    TreeOutputs finalize(int sampleToken, int maxDraftTokens) {
        // 1. Get all nodes, sort by score, and select the best candidates.
        auto allNodes = getAllNodes();
        std::sort(allNodes.begin(), allNodes.end(), [](const NodePtr& a, const NodePtr& b) {
            return a->mCumulativeLogProb > b->mCumulativeLogProb;
        });
        if (allNodes.size() > maxDraftTokens) {
            allNodes.resize(maxDraftTokens);
        }

        // 2. Sort the draft nodes by their creation order (node_id).
        std::sort(allNodes.begin(), allNodes.end(), [](const NodePtr& a, const NodePtr& b) {
            return a->mNodeId < b->mNodeId;
        });

        TreeOutputs outputs;

        // 3. Generate verifier inputs.
        std::map<int, int> nodeId2Idx;
        outputs.draftTokens.push_back(sampleToken);
        outputs.positionIds.push_back(0);
        for(size_t i = 0; i < allNodes.size(); ++i) {
            outputs.draftTokens.push_back(allNodes[i]->mTokenId);
            outputs.positionIds.push_back(allNodes[i]->mDepth + 1);
            nodeId2Idx[allNodes[i]->mNodeId] = i;
        }

        // 4. Build tree_mask.
        size_t numDraftTokens = allNodes.size();
        outputs.attentionMask.assign(numDraftTokens, std::vector<bool>(numDraftTokens, false));
        for (int i = 0; i < numDraftTokens; ++i) {
            outputs.attentionMask[i][i] = true; // self atten
            auto node = allNodes[i];
            auto parent = node->mParent.lock();
            if (parent && parent->mNodeId != -1 && nodeId2Idx.count(parent->mNodeId)) {
                int parentIdx = nodeId2Idx[parent->mNodeId];
                // Inherit parent's attention mask
                for (int j = 0; j < numDraftTokens; ++j) {
                    if (outputs.attentionMask[parentIdx][j]) {
                        outputs.attentionMask[i][j] = true;
                    }
                }
            }
        }
        // add sampleToken
        for (int i = 0; i < numDraftTokens; i++) {
            outputs.attentionMask[i].insert(outputs.attentionMask[i].begin(), true);
        }
        outputs.attentionMask.insert(outputs.attentionMask.begin(), std::vector<bool>(numDraftTokens + 1, false));
        outputs.attentionMask[0][0] = true;

        // 5. Build retrieveIndices.
        std::vector<NodePtr> leafNodes;
        for (const auto& node : allNodes) {
            bool isLeaf = true;
            for (const auto& child : node->mChildren) {
                if (nodeId2Idx.count(child->mNodeId)) {
                    isLeaf = false;
                    break;
                }
            }
            if (isLeaf) {
                leafNodes.push_back(node);
            }
        }

        for (const auto& leaf : leafNodes) {
            std::vector<int> path;
            auto curr = leaf;
            while(curr && curr->mNodeId != -1) {
                path.push_back(nodeId2Idx[curr->mNodeId] + 1);
                curr = curr->mParent.lock();
            }
            path.push_back(0);
            std::reverse(path.begin(), path.end());
            outputs.retrieveIndices.push_back(path);
        }
        return outputs;
    }

    const std::vector<std::vector<bool>>& getMask() const {
        return mMask;
    }

    std::vector<int> getIds() const {
        std::vector<int> tokenIds;
        for (auto& active : mActives) {
            tokenIds.push_back(active->mTokenId);
        }
        return tokenIds;
    }

    std::string toString(const std::function<std::string(int)>& decoder) const {
        if (mRoot->mChildren.empty()) {
            return "<Empty Tree>\n";
        }
        std::ostringstream oss;
        oss << "Tree Structure (token, cumulative_log_prob):\n";
        toStringRecursive(oss, mRoot, "", decoder);
        return oss.str();
    }
private:
    int mTopK, mCounter;
    NodePtr mRoot;
    std::vector<NodePtr> mActives;
    std::vector<std::vector<bool>> mMask;
    const int* mD2tPtr;

    int d2t(int token) {
        if (mD2tPtr) {
            return token + mD2tPtr[token];
        }
        return token;
    }

    std::vector<NodePtr> getAllNodes() {
        std::vector<NodePtr> all_nodes;
        std::queue<NodePtr> q;
        for (const auto& child : mRoot->mChildren) {
            q.push(child);
        }
        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            all_nodes.push_back(node);
            for (const auto& child : node->mChildren) {
                q.push(child);
            }
        }
        return all_nodes;
    }

    void toStringRecursive(std::ostringstream& oss, const NodePtr& node, const std::string& prefix, const std::function<std::string(int)>& decoder) const {
        if (node->mChildren.empty()) {
            return;
        }
        for (size_t i = 0; i < node->mChildren.size(); ++i) {
            const auto& child = node->mChildren[i];
            bool is_last = (i == node->mChildren.size() - 1);
            oss << prefix << (is_last ? "└── " : "├── ");
            std::string tokenStr = decoder(child->mTokenId);
            oss << "'" << tokenStr << "' (ID:" << child->mNodeId << ", CLogP:" << child->mCumulativeLogProb << ")";
            if (std::find(mActives.begin(), mActives.end(), child) != mActives.end()) {
                oss << "\033[1;32m*\033[0m";
            }
            oss << "\n";
            toStringRecursive(oss, child, prefix + (is_last ? "    " : "│   "), decoder);
        }
    }
};

} // namespace Transformer
} // namespace MNN
#endif // TOKENTREE_HPP