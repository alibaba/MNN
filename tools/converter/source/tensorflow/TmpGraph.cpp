//
//  TmpGraph.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TmpGraph.hpp"
#include <set>
#include "OpMapper.hpp"
#include "TfUtils.hpp"
#include "logkit.h"

TmpNode::TmpNode() : opName(), opType(), tfNode(nullptr), isCovered(false), isDelete(false), leftInEdges(0) {
}

TmpNode::~TmpNode(){};

std::string TmpNode::DebugString() const {
    return tfNode->DebugString();
}

TmpGraph::TmpGraph(const tensorflow::GraphDef &tfGraph) : _tfGraph(tfGraph) {
}

TmpGraph::~TmpGraph() {
    for (auto &it : tmpNodeMap) {
        delete it.second;
    }
    tmpNodeMap.clear();
}

bool TmpGraph::_allOpSupported() {
    std::set<std::string> notSupportedOp;

    for (auto &it : this->tmpNodeMap) {
        if (!it.second->isCovered && !it.second->isDelete) {
            const auto &itOpMape = tfOp2MNNOp.find(it.second->opType);
            if (itOpMape == tfOp2MNNOp.end()) {
                // Not supported
                notSupportedOp.insert(it.second->opType);
            }
        }
    }

    if (notSupportedOp.empty()) {
        return true;
    } else {
        // DLOG(INFO) << "MNN NOT_SUPPORTED_OP: "
        //               "======> ";
        // for (auto &it : notSupportedOp) {
        //     std::cout << it << ", ";
        // }
        // std::cout << "<======" << std::endl;
        // return false;
        std::string errorMessage = "\n\n===========This Model Has NOT_SUPPORTED_OP===========!!!\n";
        errorMessage += "\nMNN NOT_SUPPORTED_OP: ======>[ ";
        for (auto &it : notSupportedOp) {
            errorMessage += it;
            errorMessage += ", ";
        }
        errorMessage += " ]<======\n\n";
        DLOG(FATAL) << errorMessage;
        return false;
    }
}

void TmpGraph::_getTmpNodeMapAndConnection() {
    const int node_count = _tfGraph.node_size();

    // build the temp graph --> nodeMap
    for (int i = 0; i < node_count; i++) {
        const tensorflow::NodeDef &tfNode = _tfGraph.node(i);

        auto *tempNode   = new TmpNode();
        tempNode->opName = tfNode.name();
        tempNode->opType = tfNode.op();
        tempNode->tfNode = &tfNode;

        tmpNodeMap.insert(std::make_pair(tempNode->opName, tempNode));
    }

    // set inedge(node names) and outedge(node name) to the node
    for (int i = 0; i < node_count; i++) {
        const tensorflow::NodeDef &tfNode = _tfGraph.node(i);
        const std::string curNodeName     = tfNode.name();
        int inputSize                     = tfNode.input_size();
        TmpNode *curNode                  = this->_getTmpNode(curNodeName);
        for (int j = 0; j < inputSize; j++) {
            std::string inputName = tfNode.input(j); // may be input or input:0 or input:1
            // delete the name that has "^"
            inputName        = inputName.substr(inputName.find("^") + 1, inputName.size());
            TmpNode *srcNode = this->_getTmpNode(inputName);
            // make node connection
            this->_makeConnection(srcNode, curNode, inputName, curNodeName);
        }
    }
}

int TmpGraph::buildGraph() {
// firstly optimize tensorflow model
#if TFMODEL_OPTIMIZE
    _optimizeTfModel();
#endif

    _getTmpNodeMapAndConnection();

    // get input node(inEdges.size() == 0)
    this->_getInputNodes();

    if (_hasContinuousConstantNode()) {
        std::cout << "\n***********************" << std::endl;
        std::cout << "Strongly Recommended: Apply Tensorflow Tool [graph transform] firstly!!! ==> fold_constants"
                  << std::endl;
        std::cout << "***********************\n" << std::endl;
    }

    // get the ops that is in right order(some model whose ops are not saved as in
    // right order)
    this->_getOpsInorder(this->inputNodes);

    // delete not used node, set some Const node to isCovered
    this->_genMinGraph();

    if (!(this->_allOpSupported())) {
        DLOG(FATAL) << "===========This Model Has "
                       "NOT_SUPPORTED_OP===========!!!";
    }

    // set in and out tensor names
    const int node_count = _tfGraph.node_size();
    for (int i = 0; i < node_count; i++) {
        const tensorflow::NodeDef &tf_node = _tfGraph.node(i);
        TmpNode *current_node              = this->_getTmpNode(tf_node.name());

        const int input_size = tf_node.input_size();
        for (int j = 0; j < input_size; j++) {
            std::string input_name = tf_node.input(j);
            // delete the name that has "^"
            input_name = input_name.substr(input_name.find("^") + 1, input_name.size());
            // input name is tensor name not the node name
            TmpNode *parent_node = this->_getTmpNode(input_name);
            // const node(const-->node)
            if (parent_node->isCovered) {
                continue;
            }
            while (parent_node->isDelete) {
                parent_node = this->_getTmpNode(parent_node->inEdges[0]);
                input_name  = parent_node->opName;
            }
            // const node(const-->Indentity-->node)
            if (parent_node->isCovered) {
                continue;
            }
            if (!current_node->isDelete) {
                this->_setInOutTensorsName(parent_node, current_node, input_name);
            }
        }
    }

    return 0;
}

// sampleNode's input tensors
// input:0, input:1, input:2...
// output tensor names: input input:1 intput:2...
int TmpGraph::_setInOutTensorsName(TmpNode *parentNode, TmpNode *curNode, std::string inputName) {
    const std::string inputRealName = inputName.substr(0, inputName.find(":"));
    DCHECK(inputRealName == parentNode->opName)
        << "Input Tensor ERROR!!! ===> " << inputRealName << "--> " << parentNode->opName;
    // find the ":"
    const std::string::size_type position = inputName.find(":");
    int tensorIndex                       = -1;
    if (position != std::string::npos) { // found
        tensorIndex = std::stoi(inputName.substr(position + 1, inputName.size()).c_str());
        // input:0
        if (tensorIndex == 0) {
            this->_setOuputTensorsName(parentNode->outTensors, inputRealName, 0); // input:0 --> input
            curNode->inTensors.push_back(inputRealName); // input tensor name is also "input", not "input:0"
        } else {
            this->_setOuputTensorsName(parentNode->outTensors, inputName,
                                       tensorIndex); // input:1 --> input:1, input:2 --> input:2
            curNode->inTensors.push_back(inputName); // intput:1 --> input:1
        }
    } else {
        this->_setOuputTensorsName(parentNode->outTensors, inputName, 0); // input --> input
        curNode->inTensors.push_back(inputName);                          // input --> input
    }

    return 0;
}

int TmpGraph::_setOuputTensorsName(std::vector<std::string> &tensorVector, std::string inputName, int index) {
    const int tensorsNums = tensorVector.size();
    if (index >= tensorsNums) {
        tensorVector.resize(index + 1);
    }
    tensorVector[index] = inputName;
    return 0;
}

int TmpGraph::_pushNoReaptedItem(std::vector<std::string> &tensorNames, const std::string item) {
    std::vector<std::string>::iterator it = tensorNames.begin();
    while (it != tensorNames.end()) {
        if (item == *it) {
            return -1; // item in tensor names
        }
        it++;
    }
    if (it == tensorNames.end()) {
        tensorNames.push_back(item);
    }

    return 0;
}

int TmpGraph::_makeConnection(TmpNode *srcNode, TmpNode *dstNode, const std::string srcName,
                              const std::string dstName) {
    // node1, node2
    this->_pushNoReaptedItem(srcNode->outEdges, dstName);

    // in case: node2's input is : node1:0, node1:1
    // const std::string srcNameReal = srcName.substr(0, srcName.find(":"));
    const std::string srcNameReal = TFModelOptimizer::NodeNameFromInput(srcName);
    this->_pushNoReaptedItem(dstNode->inEdges, srcNameReal);

    return 0;
}

TmpNode *TmpGraph::_getTmpNode(const std::string &nodeName) {
    // sometimes the input of node is tensor name, not the node name(image:0 ==>
    // image)
    // std::string inputNameReal = nodeName.substr(0, nodeName.find(":"));
    // delete the name has "^"
    // inputNameReal = inputNameReal.substr(inputNameReal.find("^") + 1, inputNameReal.size());
    const auto inputNameReal = TFModelOptimizer::NodeNameFromInput(nodeName);
    const auto &it           = tmpNodeMap.find(inputNameReal);
    if (it != tmpNodeMap.end()) {
        return it->second;
    } else {
        DLOG(ERROR) << "Check The Node Name ===> [ " << nodeName << " ]";
    }
    return nullptr;
}

void TmpGraph::_genMinGraph() {
    // get the min graph ---> Const,Identity
    for (auto &it : tmpNodeMap) {
        std::string typeOp  = it.second->opType;
        TmpNode *curNode    = it.second;
        TmpNode *parentNode = nullptr;
        TmpNode *sonNode    = nullptr;

        if (typeOp == "Identity" || typeOp == "StopGradient") {
            curNode->isDelete = true;
            // asume the Identity has one input
            DCHECK(curNode->inEdges.size() == 1) << "Identity's input node num. != 1 [ " << curNode->opName << " ]";
            std::string parentName = curNode->inEdges[0];
            parentNode             = this->_getTmpNode(parentName);
            // delete Identity node and connect again
            for (int i = 0; i < curNode->outEdges.size(); i++) {
                const std::string sonName = curNode->outEdges[i];
                sonNode                   = this->_getTmpNode(sonName);
                this->_changInOutName(parentNode->outEdges, sonName, curNode->opName);
                this->_changInOutName(sonNode->inEdges, parentName, curNode->opName);
            }
        }
        // next node is BiasAdd
        else if (typeOp == "Conv2D" || typeOp == "DepthwiseConv2dNative" || typeOp == "Conv2DBackpropInput") {
            parentNode = this->_getTmpNode(curNode->inEdges[1]);

            if (parentNode->opType == "Identity") {
                parentNode = this->_getTmpNode(parentNode->inEdges[0]);
            }

            if (parentNode->opType == "Const") { // weight
                parentNode->isCovered = true;
            }

            if (curNode->outEdges.size() != 1)
                continue;                                      // next node cann't be BiasAdd
            sonNode = this->_getTmpNode(curNode->outEdges[0]); // BiasAdd or Add Node
            if (sonNode->opType != "BiasAdd" && sonNode->opType != "Add") {
                continue;
            }

            // ---> next node must be BiasAdd(Add) which should be merged in Conv node
            // change the inEdges of the node that is at next position of BiasAdd
            // (conv->biasadd->node1)
            for (int i = 0; i < sonNode->outEdges.size(); i++) {
                TmpNode *biasNodeSon = this->_getTmpNode(sonNode->outEdges[i]);
                this->_changInOutName(biasNodeSon->inEdges, curNode->opName, sonNode->opName);
                // delete the BiasAdd Node, set the output node of BiasAdd node to Conv
                // node's output
                this->_changInOutName(curNode->outEdges, sonNode->outEdges[i], sonNode->opName);
            }

            TmpNode *biasVariable = this->_getTmpNode(sonNode->inEdges[1]); // bias(Const)

            if (biasVariable->opType == "Identity") {
                biasVariable = this->_getTmpNode(biasVariable->inEdges[0]);
            }
            if (biasVariable->opType != "Const") {
                continue;
            }
            biasVariable->isCovered = true;                   // Const op not convertered to MemoryData
            curNode->inEdges.push_back(biasVariable->opName); // put bias node to conv inEdges
            sonNode->isDelete = true;                         // delete BiasAdd op
        }
        // BatchNorm
        else if (typeOp == "FusedBatchNorm" || typeOp == "SpaceToBatchND" || typeOp == "BatchToSpaceND") {
            for (int i = 1; i < curNode->inEdges.size(); i++) {
                TmpNode *inputNode = this->_getTmpNode(curNode->inEdges[i]);

                if (inputNode->opType == "Identity") {
                    inputNode = this->_getTmpNode(inputNode->inEdges[0]);
                }

                DCHECK(inputNode->opType == "Const") << "FusedBatchNorm|SpaceToBatchND Lack Const Tensor";
                inputNode->isCovered = true;
            }
        } else if (typeOp == "Reshape") {
            DCHECK(curNode->inEdges.size() == 2) << "Reshape Should Have Two Input!!! ===> " << curNode->opName;
            TmpNode *shapeNode = this->_getTmpNode(curNode->inEdges[1]);
            // DCHECK(shapeNode->opType == "Const") << "Reshape  Now Only Support
            // Const Shape Input!!! ===> " << curNode->opName;
            if (shapeNode->opType == "Const") {
                shapeNode->isCovered = true;
            }
        }

        else if (typeOp == "ConcatV2" || typeOp == "Concat") {
            TmpNode *constAxisInput = nullptr;
            if ("ConcatV2" == typeOp) {
                constAxisInput = this->_getTmpNode(curNode->inEdges.back());
            } else {
                constAxisInput = this->_getTmpNode(curNode->inEdges[0]);
            }
            if (constAxisInput->opType == "Identity") {
                constAxisInput = this->_getTmpNode(constAxisInput->inEdges[0]);
            }
            DCHECK(constAxisInput->opType == "Const") << "Concat Have no axis Input!!! => " << curNode->opName;
            constAxisInput->isCovered = true;
        } else if (typeOp == "Split") {
            TmpNode *dimInput = this->_getTmpNode(curNode->inEdges[0]);
            if (dimInput->opType == "Identity") {
                dimInput = this->_getTmpNode(dimInput->inEdges[0]);
            }
            DCHECK(dimInput->opType == "Const") << "Split Have no axis Input!!! => " << curNode->opName;
            dimInput->isCovered = true;
        } else if (typeOp == "ResizeBilinear" || typeOp == "Mean" || typeOp == "Sum" || typeOp == "Max" ||
                   typeOp == "Min" || typeOp == "Prod" || typeOp == "ArgMax" || typeOp == "Moments") {
            // size input
            parentNode = this->_getTmpNode(curNode->inEdges[1]);
            // const op read
            if (parentNode->opType == "Identity") {
                parentNode = this->_getTmpNode(parentNode->inEdges[0]);
            }
            if (parentNode->opType == "Const") {
                parentNode->isCovered = true;
            }
        } else if (typeOp == "SplitV") {
            DCHECK(3 == curNode->inEdges.size()) << "SplitV should have three inputs";
            for (int i = 1; i < 3; ++i) {
                auto inNodeSplit = this->_getTmpNode(curNode->inEdges[i]);
                if (inNodeSplit->opType == "Identity") {
                    inNodeSplit = this->_getTmpNode(inNodeSplit->inEdges[0]);
                }
                DCHECK("Const" == inNodeSplit->opType);
                inNodeSplit->isCovered = true;
            }
        } else if (typeOp == "InstanceNorm") {
            DCHECK(4 == curNode->inEdges.size()) << "InstanceNorm should have four inputs";
            for (int i = 1; i < 3; ++i) {
                auto inputNode = _getTmpNode(curNode->inEdges[i]);
                if (inputNode->opType == "Identity") {
                    inputNode = _getTmpNode(inputNode->inEdges[0]);
                }
                DCHECK("Const" == inputNode->opType);
                inputNode->isCovered = true;
            }
        } else if (typeOp == "RNNSequenceGRU") {
            for (int i = 1; i < curNode->inEdges.size(); ++i) {
                auto inputNode = _getTmpNode(curNode->inEdges[i]);
                if (inputNode->opType == "Identity") {
                    inputNode = _getTmpNode(inputNode->inEdges[0]);
                }
                DCHECK("Const" == inputNode->opType);
                inputNode->isCovered = true;
            }
        }
    }
}

void TmpGraph::_changInOutName(std::vector<std::string> &inOutEdges, std::string name, std::string deleteName) {
    std::vector<std::string>::iterator it = inOutEdges.begin();
    while (it != inOutEdges.end()) {
        if (deleteName == *it) {
            *it = name;
            return;
        }
        it++;
    }
    if (it == inOutEdges.end()) {
        inOutEdges.push_back(name);
    }
}

int TmpGraph::_getOpsInorder(const std::vector<std::string> inputNodes) {
    this->opsInOrder.clear();
    this->opsInOrder = inputNodes;
    int index        = 0;
    while (index < this->opsInOrder.size()) {
        TmpNode *currentNode = this->_getTmpNode(this->opsInOrder[index]);
        for (int i = 0; i < currentNode->outEdges.size(); i++) {
            TmpNode *nextNode = this->_getTmpNode(currentNode->outEdges[i]);
            nextNode->leftInEdges--;
            if (nextNode->leftInEdges == 0) {
                this->_pushNoReaptedItem(this->opsInOrder, currentNode->outEdges[i]);
            }
        }
        index++;
    }

    return 0;
}

void TmpGraph::_getInputNodes() {
    this->inputNodes.clear();
    const int nodeCount = this->_tfGraph.node_size();
    for (int i = 0; i < nodeCount; i++) {
        const tensorflow::NodeDef &tfNode = this->_tfGraph.node(i);
        TmpNode *tempNode                 = this->_getTmpNode(tfNode.name());
        tempNode->leftInEdges             = tempNode->inEdges.size();
        if (tempNode->inEdges.size() == 0) {
            this->_pushNoReaptedItem(this->inputNodes, tfNode.name());
        }
    }
}

bool TmpGraph::_hasContinuousConstantNode() {
    std::map<std::string, bool> visited;
    for (auto &item : tmpNodeMap) {
        if (std::find(inputNodes.begin(), inputNodes.end(), item.first) != inputNodes.end()) {
            visited[item.first] = true;
        }
        visited[item.first] = false;
    }
    for (const auto &node_name : inputNodes) {
        auto cur_node = _getTmpNode(node_name);
        for (const auto &out : cur_node->outEdges) {
            auto son_node = _getTmpNode(out);
            if (visited.at(son_node->opName) || son_node->opType == "Identity") {
                continue;
            }
            visited[son_node->opName] = true;
            bool isConstantNode       = true;

            for (const auto &input : son_node->inEdges) {
                auto parenr_node = _getTmpNode(input);
                if (parenr_node->opType != "Const") {
                    isConstantNode = false;
                    break;
                }
            }
            if (isConstantNode) {
                return true;
            }
        }
    }
    return false;
}

int TmpGraph::_optimizeTfModel() {
    // using namespace TFModelOptimizer;
    // TransformRegistry* transformRegistry = GetTransformRegistry();
    // for(auto transformFunc : *transformRegistry){
    //     TransformFuncContext context;
    //     GraphDef transformed_graph_def;
    //     auto func = transformFunc.second;
    //     func(*_tfGraph, context, &transformed_graph_def);
    //     *_tfGraph = transformed_graph_def;
    // }
    TFModelOptimizer::TransformFuncContext context;
    context.params["op"].push_back("Identity");
    context.params["op"].push_back("StopGradient");
    tensorflow::GraphDef transformed_graph_def;

    TFModelOptimizer::RemoveNodes(_tfGraph, context, &transformed_graph_def);
    _tfGraph = transformed_graph_def;

    TFModelOptimizer::FoldBatchNormsAlgebraic(_tfGraph, context, &transformed_graph_def);
    _tfGraph = transformed_graph_def;

    TFModelOptimizer::FoldMoments(_tfGraph, context, &transformed_graph_def);
    _tfGraph = transformed_graph_def;

    TFModelOptimizer::ResolveRNNGRUCell(_tfGraph, context, &transformed_graph_def);
    _tfGraph = transformed_graph_def;

    TFModelOptimizer::FuseConvPad(_tfGraph, context, &transformed_graph_def);
    _tfGraph = transformed_graph_def;

    TFModelOptimizer::FuseRelu6(_tfGraph, context, &transformed_graph_def);
    _tfGraph = transformed_graph_def;

    return 0;
}
