//
//  DirectedAcyclicGraphTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <sstream>
#include <string>
#include "core/DirectedAcyclicGraph.hpp"
#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"

using namespace MNN;

class OPCustom {
public:
    OPCustom(string n) {
        name = n;
    };
    virtual ~OPCustom(){
        // MNN_PRINT("OPCustom free\n");
    };

public:
    void setName(string n) {
        name = n;
    }
    string getName() {
        return name;
    }

private:
    string name;
};

class OPCustomNodeDef : public NodeDef<shared_ptr<OPCustom>> {
public:
    OPCustomNodeDef(string name) {
        this->name = name;
    }

public:
    void setName(string n) {
        this->name = n;
    }

public:
    virtual shared_ptr<Node<shared_ptr<OPCustom>>> makeNode() override {
        shared_ptr<Node<shared_ptr<OPCustom>>> ptr = make_shared<Node<shared_ptr<OPCustom>>>();
        shared_ptr<OPCustom> op                    = make_shared<OPCustom>(name);
        ptr->setData(op);
        return ptr;
    }

private:
    string name;
};

static int stringCounter(const string& str, const string& sub) {
    int num = 0;
    for (size_t i = 0; (i = str.find(sub, i)) != string::npos; num++, i++) {
        // do nothing
    }
    return num;
}

static bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static bool startsWith(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

/* *
 * input A->B->C->D expect output A->B->C->D return true
 * smart pointer use_count == 2
 * */
static void TestMemoryLeak() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);
    graph->AddEdge(A, B);
    graph->AddEdge(B, C);
    graph->AddEdge(C, D);
    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    graph.reset();
    A.reset();
    B.reset();
    C.reset();
    D.reset();

    stringstream ss;
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->" << op.use_count() << "\t";
    }

    const string rel_str(ss.str());
    const string exp_str = "A->2\tB->2\tC->2\tD->2\t";
    const int exp_val    = exp_str.compare(rel_str);
    if ((exp_val != 0) || (!ok)) {
        MNN_ERROR("TestMemoryLeak expect '%s,ok=1' output is %s,ok=%d\n", exp_str.c_str(), rel_str.c_str(), ok);
    }
}

/* *
 * input A C->B D expect output A->C->B->D or A->D->C->B or D->A->C->B or C->B->A->D or C->B->D->A return true
 * input A C->B D->B expect output A->C->D->B or C->D->B->A return true
 * input C->B D->B C->A expect output C->A->D->B or  D->C->A->B or D->C->B->A return true
 * input C->B D->B C->A D->C expect output D->C->A->B or  D->C->B->A return true
 * input C->B D->B C->A D->C A->C expect return false
 * */
static void TestPostOrderSinglePoint() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);
    graph->AddEdge(C, B);
    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    string rel_str(ss.str());
    string exp_str  = "A->C->B->D->";
    string exp_str2 = "A->D->C->B->";
    string exp_str3 = "D->A->C->B->";
    string exp_str4 = "C->B->D->A->";
    string exp_str5 = "C->B->A->D->";
    int exp_val     = exp_str.compare(rel_str);
    if (0 != exp_val) {
        exp_val = exp_str2.compare(rel_str);
    }
    if (0 != exp_val) {
        exp_val = exp_str3.compare(rel_str);
    }
    if (0 != exp_val) {
        exp_val = exp_str4.compare(rel_str);
    }
    if (0 != exp_val) {
        exp_val = exp_str5.compare(rel_str);
    }
    if ((!ok) || (0 != exp_val)) {
        MNN_ERROR("TestPostOrderSinglePoint expect 'A->C->B->D,ok=1' output is %s,ok=%d\n", rel_str.c_str(), ok);
    }

    graph->AddEdge(D, B);
    ok = graph->GetPostOrder(order);
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    rel_str  = ss.str();
    exp_str  = "A->D->C->B->";
    exp_str2 = "A->C->D->B->";
    exp_str3 = "C->D->B->A->";
    exp_val  = exp_str.compare(rel_str);
    if (0 != exp_val) {
        exp_val = exp_str2.compare(rel_str);
    }
    if (0 != exp_val) {
        exp_val = exp_str3.compare(rel_str);
    }
    if ((!ok) || (0 != exp_val)) {
        MNN_ERROR("TestPostOrderSinglePoint expect 'A->C->D->B or A->D->C->B,ok=1' output is %s,ok=%d\n",
                  rel_str.c_str(), ok);
    }

    graph->AddEdge(C, A);
    ok = graph->GetPostOrder(order);
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    rel_str  = ss.str();
    exp_str  = "C->A->D->B->";
    exp_str2 = "D->C->A->B->";
    exp_str3 = "D->C->B->A->";
    exp_val  = exp_str.compare(rel_str);
    if (0 != exp_val) {
        exp_val = exp_str2.compare(rel_str);
    }
    if (0 != exp_val) {
        exp_val = exp_str3.compare(rel_str);
    }
    if ((!ok) || (0 != exp_val)) {
        MNN_ERROR("TestPostOrderSinglePoint expect 'C->A->D->B,ok=1' output is %s,ok=%d\n", rel_str.c_str(), ok);
    }

    graph->AddEdge(D, C);
    ok = graph->GetPostOrder(order);
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    rel_str  = ss.str();
    exp_str  = "D->C->A->B->";
    exp_str2 = "D->C->B->A->";
    exp_val  = exp_str.compare(rel_str);
    if (0 != exp_val) {
        exp_val = exp_str2.compare(rel_str);
    }
    if ((!ok) || (0 != exp_val)) {
        MNN_ERROR("TestPostOrderSinglePoint expect 'D->C->A->B or D->C->B->A,ok=1' output is %s,ok=%d\n",
                  rel_str.c_str(), ok);
    }

    /*cycle*/
    graph->AddEdge(A, C);

    ok = graph->GetPostOrder(order);
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }
    if (false != ok) {
        MNN_ERROR("TestPostOrderSinglePoint cycle expect 'ok=0' output is %s,ok=%d\n", ss.str().c_str(), ok);
    }
}

/* *
 * input A->B->C->D expect output A->B->C->D return true
 * input A->B->C->D->A expect return false
 * */
static void TestPostOrder() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);
    graph->AddEdge(A, B);
    graph->AddEdge(B, C);
    graph->AddEdge(C, D);
    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    const string rel_str(ss.str());
    const string exp_str = "A->B->C->D->";
    const int exp_val    = exp_str.compare(rel_str);
    if ((!ok) || (0 != exp_val)) {
        MNN_ERROR("TestPostOrder expect 'A->B->C->D,ok=1' output is %s,ok=%d\n", rel_str.c_str(), ok);
    }

    /*cycle*/
    graph->AddEdge(D, B);
    ok = graph->GetPostOrder(order);
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }
    if (false != ok) {
        MNN_ERROR("TestPostOrder cycle expect 'ok=0' output is %s,ok=%d\n", ss.str().c_str(), ok);
    }
}

/* *
 * input A->B->C->D expect output A->B->C->D return true
 * input A->B->C->D->A expect return false
 * */
static void TestPostOrderDiffInputs() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);
    graph->AddEdge(C, D);
    graph->AddEdge(B, C);
    graph->AddEdge(A, B);
    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }
    const string rel_str(ss.str());
    const string exp_str = "A->B->C->D->";
    const int exp_val    = exp_str.compare(rel_str);
    if ((!ok) || (0 != exp_val)) {
        MNN_ERROR("TestPostOrderDiffInputs expect 'A->B->C->D,ok=1' output is %s,ok=%d\n", rel_str.c_str(), ok);
    }

    /*cycle*/
    graph->AddEdge(D, B);
    ok = graph->GetPostOrder(order);
    ss.str("");
    ss.clear();
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }
    if (false != ok) {
        MNN_ERROR("TestPostOrderDiffInputs cycle expect 'ok=0' output is %s,ok=%d\n", ss.str().c_str(), ok);
    }
}

/* *
 * input A B C D expect return true do'nt care order,only contain A B C D
 * */
static void TestPostOrderAllSingle() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);
    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    const string rel_str(ss.str());
    const string exp_str1 = "A->";
    const string exp_str2 = "B->";
    const string exp_str3 = "C->";
    const string exp_str4 = "D->";
    const int exp_val1    = stringCounter(rel_str, exp_str1);
    const int exp_val2    = stringCounter(rel_str, exp_str2);
    const int exp_val3    = stringCounter(rel_str, exp_str3);
    const int exp_val4    = stringCounter(rel_str, exp_str4);
    const int exp_len     = (int)(exp_str1.length() + exp_str2.length() + exp_str3.length() + exp_str4.length());
    if ((exp_val1 != 1) || (exp_val2 != 1) || (exp_val3 != 1) || (exp_val4 != 1) || (!ok) ||
        (rel_str.length() != exp_len)) {
        MNN_ERROR("TestPostOrderAllSingle expect only contain 'A B C D,ok=1' ignore order output is %s,ok=%d\n",
                  rel_str.c_str(), ok);
    }
}

/* *
 * input A->B A->C A->D expect return true and A is first
 * */
static void TestPostOrderAllFromOne() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);

    graph->AddEdge(A, D);
    graph->AddEdge(A, C);
    graph->AddEdge(A, B);

    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    const string rel_str(ss.str());
    const string exp_str1 = "A->";
    const string exp_str2 = "B->";
    const string exp_str3 = "C->";
    const string exp_str4 = "D->";
    const int exp_val1    = stringCounter(rel_str, exp_str1);
    const int exp_val2    = stringCounter(rel_str, exp_str2);
    const int exp_val3    = stringCounter(rel_str, exp_str3);
    const int exp_val4    = stringCounter(rel_str, exp_str4);
    const int exp_len     = (int)(exp_str1.length() + exp_str2.length() + exp_str3.length() + exp_str4.length());
    const bool exp_val    = startsWith(rel_str, exp_str1);

    if ((exp_val1 != 1) || (exp_val2 != 1) || (exp_val3 != 1) || (exp_val4 != 1) || (!ok) ||
        (rel_str.length() != exp_len) || (!exp_val)) {
        MNN_ERROR("TestPostOrderAllFromOne expect A is first output is %s,ok=%d\n", rel_str.c_str(), ok);
    }
}

/* *
 * input B->A C->A D->A expect return true and A is last
 * */
static void TestPostOrderAllToOne() {
    OPCustomNodeDef def("A");
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    shared_ptr<Node<shared_ptr<OPCustom>>> A = graph->AddNode(def);
    def.setName("B");
    shared_ptr<Node<shared_ptr<OPCustom>>> B = graph->AddNode(def);
    def.setName("C");
    shared_ptr<Node<shared_ptr<OPCustom>>> C = graph->AddNode(def);
    def.setName("D");
    shared_ptr<Node<shared_ptr<OPCustom>>> D = graph->AddNode(def);

    graph->AddEdge(D, A);
    graph->AddEdge(C, A);
    graph->AddEdge(B, A);

    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    const string rel_str(ss.str());
    const string exp_str1 = "A->";
    const string exp_str2 = "B->";
    const string exp_str3 = "C->";
    const string exp_str4 = "D->";
    const int exp_val1    = stringCounter(rel_str, exp_str1);
    const int exp_val2    = stringCounter(rel_str, exp_str2);
    const int exp_val3    = stringCounter(rel_str, exp_str3);
    const int exp_val4    = stringCounter(rel_str, exp_str4);
    const int exp_len     = (int)(exp_str1.length() + exp_str2.length() + exp_str3.length() + exp_str4.length());
    const bool exp_val    = endsWith(rel_str, exp_str1);
    if ((exp_val1 != 1) || (exp_val2 != 1) || (exp_val3 != 1) || (exp_val4 != 1) || (!ok) ||
        (rel_str.length() != exp_len) || (!exp_val)) {
        MNN_ERROR("TestPostOrderAllToOne expect A is last output is %s,ok=%d\n", rel_str.c_str(), ok);
    }
}

/* *
 * expect return true
 * */
static void TestPostOrderEmpty() {
    unique_ptr<DirectedAcyclicGraph<shared_ptr<OPCustom>>> graph(new DirectedAcyclicGraph<shared_ptr<OPCustom>>());
    vector<shared_ptr<Node<shared_ptr<OPCustom>>>> order;
    bool ok = graph->GetPostOrder(order);
    stringstream ss;
    for (shared_ptr<Node<shared_ptr<OPCustom>>> op : order) {
        string name = op->getData()->getName();
        ss << name << "->";
    }

    const string rel_str(ss.str());
    if ((!ok) || (rel_str.length() != 0)) {
        MNN_ERROR("TestPostOrderEmpty expect 'ok=1',%s output is ok=%d\n", rel_str.c_str(), ok);
    }
}

class DirectedAcyclicGraphTest : public MNNTestCase {
public:
    virtual bool run();
    DirectedAcyclicGraphTest() {
    }
    virtual ~DirectedAcyclicGraphTest() {
    }
};

bool DirectedAcyclicGraphTest::run() {
    TestPostOrder();
    TestPostOrderSinglePoint();
    TestMemoryLeak();
    TestPostOrderDiffInputs();
    TestPostOrderAllSingle();
    TestPostOrderAllFromOne();
    TestPostOrderAllToOne();
    TestPostOrderEmpty();
    return true;
}

MNNTestSuiteRegister(DirectedAcyclicGraphTest, "engine/DirectedAcyclicGraph");
