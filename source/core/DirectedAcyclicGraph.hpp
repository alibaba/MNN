//
//  DirectedAcyclicGraph.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
namespace MNN {
template <typename T>
class Node;

template <typename T>
class Edge {
public:
    void setSrc(shared_ptr<Node<T> > node) {
        this->srcNode = weak_ptr<Node<T> >(node);
    }

    void setDst(shared_ptr<Node<T> > node) {
        this->dstNode = weak_ptr<Node<T> >(node);
    }

    const weak_ptr<Node<T> > getSrc() {
        return srcNode;
    }

    const weak_ptr<Node<T> > getDst() {
        return dstNode;
    }

private:
    weak_ptr<Node<T> > srcNode;
    weak_ptr<Node<T> > dstNode;
};

template <typename T>
class Node {
public:
    void addInEdge(shared_ptr<Edge<T> > edge) {
        this->inEdges.insert(edge);
    }

    void addOutEdge(shared_ptr<Edge<T> > edge) {
        this->outEdges.insert(edge);
    }

    const unordered_set<shared_ptr<Edge<T> > > getInEdges() {
        return inEdges;
    }

    const unordered_set<shared_ptr<Edge<T> > > getOutEdges() {
        return outEdges;
    }

    const int getInEdgesCount() {
        return (int)inEdges.size();
    }

    void setData(T d) {
        this->data = d;
    }

    T getData() {
        return data;
    }

private:
    T data;
    unordered_set<shared_ptr<Edge<T> > > inEdges;
    unordered_set<shared_ptr<Edge<T> > > outEdges;
};

template <typename T>
class NodeDef {
public:
    virtual shared_ptr<Node<T> > makeNode() {
        return make_shared<Node<T> >();
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
    shared_ptr<Node<T> > AddNode(NodeDef<T>& node_def) {
        shared_ptr<Node<T> > node = node_def.makeNode();
        nodes.insert(make_pair(node, nodes.size()));
        return node;
    }

    /**
     * Adds an edge that connects `source` input of
     * `dest` and returns it.
     */
    const shared_ptr<Edge<T> > AddEdge(shared_ptr<Node<T> > source, shared_ptr<Node<T> > dest) {
        shared_ptr<Edge<T> > edge = make_shared<Edge<T> >();
        edge->setSrc(source);
        edge->setDst(dest);
        source->addOutEdge(edge);
        dest->addInEdge(edge);
        edges.insert(make_pair(edge, edges.size()));
        return edge;
    }

    /**
     * Stores in *order the post-order numbering of all nodes
     * in graph found via topological sorting.
     *
     * return true if graph does not have cycles else false .
     */
    bool GetPostOrder(vector<shared_ptr<Node<T> > >& order) {
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
    bool TopologicalSort(vector<shared_ptr<Node<T> > >& order) {
        struct TopoNode {
            shared_ptr<Node<T> > node;
            unordered_set<shared_ptr<Edge<T> > > outEdges;
        };

        unordered_map<shared_ptr<Node<T> >, unordered_set<shared_ptr<Edge<T> > > > nodesInEdges;
        /*no incoming node*/
        vector<TopoNode> noIncoming;
        typename unordered_map<shared_ptr<Node<T> >, int>::iterator iter;
        for (iter = this->nodes.begin(); iter != this->nodes.end(); iter++) {
            if (iter->first->getInEdgesCount() <= 0) {
                TopoNode tn;
                tn.node     = iter->first;
                tn.outEdges = iter->first->getOutEdges();
                noIncoming.push_back(tn);
            } else {
                nodesInEdges.insert(make_pair(iter->first, iter->first->getInEdges()));
            }
        }
        while (noIncoming.size() > 0) {
            TopoNode n = noIncoming.back();
            noIncoming.pop_back();
            order.push_back(n.node);
            for (const shared_ptr<Edge<T> >& outEdge : n.outEdges) {
                const weak_ptr<Node<T> > oNode = outEdge->getDst();
                if (!oNode.expired()) {
                    const shared_ptr<Node<T> > node = oNode.lock();
                    typename unordered_map<shared_ptr<Node<T> >, unordered_set<shared_ptr<Edge<T> > > >::iterator
                        edg_iter;
                    /*find node from nodesInEdges,and remove edge*/
                    for (edg_iter = nodesInEdges.begin(); edg_iter != nodesInEdges.end(); edg_iter++) {
                        if (edg_iter->first == node) {
                            edg_iter->second.erase(outEdge);
                            if (edg_iter->second.size() <= 0) {
                                TopoNode tn;
                                tn.node     = node;
                                tn.outEdges = node->getOutEdges();
                                noIncoming.push_back(tn);
                                nodesInEdges.erase(edg_iter);
                            }
                            break;
                        }
                    }
                    // ASSERT(edg_iter == nodes.end())
                }
            }
        }
        if (nodesInEdges.size() > 0) {
            return false;
        }
        return true;
    }

private:
    // Allocated nodes and edges.
    unordered_map<shared_ptr<Node<T> >, int> nodes;
    unordered_map<shared_ptr<Edge<T> >, int> edges;
};
} // namespace MNN
