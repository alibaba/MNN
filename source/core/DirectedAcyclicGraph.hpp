//
//  DirectedAcyclicGraph.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace MNN {
template <typename T>
class Node;

template <typename T>
class Edge {
public:
    void setSrc(std::shared_ptr<Node<T>> node) {
        srcNode = std::weak_ptr<Node<T>>(node);
    }

    void setDst(std::shared_ptr<Node<T>> node) {
        dstNode = std::weak_ptr<Node<T>>(node);
    }

    std::weak_ptr<Node<T>> getSrc() const {
        return srcNode;
    }

    std::weak_ptr<Node<T>> getDst() const {
        return dstNode;
    }

private:
    std::weak_ptr<Node<T>> srcNode;
    std::weak_ptr<Node<T>> dstNode;
};

template <typename T>
class Node {
public:
    void addInEdge(std::shared_ptr<Edge<T>> edge) {
        inEdges.insert(edge);
    }

    void addOutEdge(std::shared_ptr<Edge<T>> edge) {
        outEdges.insert(edge);
    }

    std::unordered_set<std::shared_ptr<Edge<T>>> getInEdges() const {
        return inEdges;
    }

    std::unordered_set<std::shared_ptr<Edge<T>>> getOutEdges() const {
        return outEdges;
    }

    int getInEdgesCount() const {
        return inEdges.size();
    }

    void setData(T d) {
        data = d;
    }

    T getData() const {
        return data;
    }

private:
    T data;
    std::unordered_set<std::shared_ptr<Edge<T>>> inEdges;
    std::unordered_set<std::shared_ptr<Edge<T>>> outEdges;
};

template <typename T>
class NodeDef {
public:
    virtual std::shared_ptr<Node<T>> makeNode() {
        return std::make_shared<Node<T>>();
    }
};

/**
 * A DirectedAcyclicGraph describes a set of computations that are to be
 * performed, as well as the dependencies between those
 * computations. The basic model is a DAG (directed acyclic graph)
 */
template <typename T>
class DirectedAcyclicGraph {
public:
    /**
     * Adds a new node to this graph, and returns it.
     */
    std::shared_ptr<Node<T>> AddNode(NodeDef<T>& node_def) {
        std::shared_ptr<Node<T>> node = node_def.makeNode();
        nodes.emplace(node, nodes.size());
        return node;
    }

    /**
     * Adds an edge that connects `source` input of
     * `dest` and returns it.
     */
    std::shared_ptr<Edge<T>> AddEdge(std::shared_ptr<Node<T>> source, std::shared_ptr<Node<T>> dest) {
        std::shared_ptr<Edge<T>> edge = std::make_shared<Edge<T>>();
        edge->setSrc(source);
        edge->setDst(dest);
        source->addOutEdge(edge);
        dest->addInEdge(edge);
        edges.emplace(edge, edges.size());
        return edge;
    }

    /**
     * Stores in *order the post-order numbering of all nodes
     * in graph found via topological sorting.
     *
     * return true if graph does not have cycles else false .
     */
    bool GetPostOrder(std::vector<std::shared_ptr<Node<T>>>& order) {
        order.clear();
        return TopologicalSort(order);
    }

private:
    /**
     * Kahn's algorithm
     * topological sort
     *
     *   L ← Empty list that will contain the sorted elements
     *   S ← Set of all nodes with no incoming edge
     *   while S is non-empty do
     *       remove a node n from S
     *       add n to tail of L
     *       for each node m with an edge e from n to m do
     *           remove edge e from the graph
     *           if m has no other incoming edges then
     *               insert m into S
     *  if graph has edges then
     *      return error   (graph has at least one cycle)
     *  else
     *       return L   (a topologically sorted order)
     */
    bool TopologicalSort(std::vector<std::shared_ptr<Node<T>>>& order) {
        struct TopoNode {
            std::shared_ptr<Node<T>> node;
            std::unordered_set<std::shared_ptr<Edge<T>>> outEdges;
        };

        std::unordered_map<std::shared_ptr<Node<T>>, std::unordered_set<std::shared_ptr<Edge<T>>>> nodesInEdges;
        /*no incoming node*/
        std::vector<TopoNode> noIncoming;
        for (auto iter = this->nodes.begin(); iter != this->nodes.end(); ++iter) {
            if (iter->first->getInEdgesCount() <= 0) {
                TopoNode tn;
                tn.node     = iter->first;
                tn.outEdges = iter->first->getOutEdges();
                noIncoming.push_back(tn);
            } else {
                nodesInEdges.emplace(iter->first, iter->first->getInEdges());
            }
        }

        while (!noIncoming.empty()) {
            TopoNode n = noIncoming.back();
            noIncoming.pop_back();
            order.push_back(n.node);
            for (const auto& outEdge : n.outEdges) {
                const auto oNode = outEdge->getDst();
                if (!oNode.expired()) {
                    const auto node = oNode.lock();
                    /*find node from nodesInEdges and remove edge*/
                    auto edg_iter = nodesInEdges.find(node);
                    if (edg_iter != nodesInEdges.end()) {
                        edg_iter->second.erase(outEdge);
                        if (edg_iter->second.empty()) {
                            TopoNode tn;
                            tn.node     = node;
                            tn.outEdges = node->getOutEdges();
                            noIncoming.push_back(tn);
                            nodesInEdges.erase(edg_iter);
                        }
                    }
                    // ASSERT(edg_iter == nodes.end())
                }
            }
        }
        if (!nodesInEdges.empty()) {
            return false;
        }
        return true;
    }

private:
    // Allocated nodes and edges.
    std::unordered_map<std::shared_ptr<Node<T>>, int> nodes;
    std::unordered_map<std::shared_ptr<Edge<T>>, int> edges;
};
} // namespace MNN
