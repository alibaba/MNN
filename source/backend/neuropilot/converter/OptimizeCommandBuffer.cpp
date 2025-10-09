#include "core/TensorUtils.hpp"
#include "OptimizeCommandBuffer.hpp"
namespace MNN {

OptimizeCommandBuffer::OptimizeCommandBuffer(ConvertTflite* root) {
    mRoot = root;
}
OptimizeCommandBuffer::~OptimizeCommandBuffer() {
    // Do nothing
}
static void _swapDequantAndReshape(ConvertTflite::CommandBuffer& cmdBuffer, ConvertTflite* root) {
    auto reshapeCode = root->getOpIndex(tflite::BuiltinOperator_RESHAPE);
    auto dequantCode = root->getOpIndex(tflite::BuiltinOperator_DEQUANTIZE);
    std::vector<int> reshapeIndexes;
    std::map<Tensor*, int> reshapeOutput;
    std::map<Tensor*, int> dequantOutput;
    for (int i=0; i<cmdBuffer.commands.size(); ++i) {
        if (nullptr == cmdBuffer.commands[i].op.get()) {
            continue;
        }
        if (cmdBuffer.commands[i].op->opcode_index == reshapeCode) {
            reshapeIndexes.emplace_back(i);
            if (TensorUtils::getDescribe(cmdBuffer.commands[i].outputs[0])->usage != Tensor::InsideDescribe::OUTPUT) {
                reshapeOutput.insert(std::make_pair(cmdBuffer.commands[i].outputs[0], i));
            }
            continue;
        }
        if (cmdBuffer.commands[i].op->opcode_index == dequantCode) {
            auto out = cmdBuffer.commands[i].outputs[0];
            auto des = TensorUtils::getDescribe(out);
            if (des->usage != Tensor::InsideDescribe::OUTPUT && des->useCount == 1) {
                dequantOutput.insert(std::make_pair(cmdBuffer.commands[i].outputs[0], i));
            }
            continue;
        }
    }
    // Swap Dequant And Reshape
    for (int i=0; i<reshapeIndexes.size(); ++i) {
        auto& curCmd = cmdBuffer.commands[reshapeIndexes[i]];
        auto iter = dequantOutput.find(curCmd.inputs[0]);
        if (iter == dequantOutput.end()) {
            continue;
        }
        auto& deqCmd = cmdBuffer.commands[iter->second];
        // C (DEQ)->A (RESHAPE)->B  ---> C(RESHAPE)->A (DEQ)->B
        auto A = curCmd.inputs[0];
        auto B = curCmd.outputs[0];
        auto C = deqCmd.inputs[0];
        deqCmd.inputs[0] = A;
        deqCmd.outputs[0] = B;
        curCmd.inputs[0] = C;
        curCmd.outputs[0] = A;
        TensorUtils::getDescribe(A)->quantAttr = TensorUtils::getDescribe(C)->quantAttr;
        TensorUtils::getDescribe(A)->applyQuant = TensorUtils::getDescribe(C)->applyQuant;
        reshapeIndexes[i] = iter->second;
        std::swap(cmdBuffer.commands[reshapeIndexes[i]], cmdBuffer.commands[iter->second]);
    }
}
static void _removeDupReshape(ConvertTflite::CommandBuffer& cmdBuffer, ConvertTflite* root) {
    auto reshapeCode = root->getOpIndex(tflite::BuiltinOperator_RESHAPE);
    // Find reshape
    std::vector<int> reshapeIndexes;
    std::map<Tensor*, int> reshapeOutput;
    for (int i=0; i<cmdBuffer.commands.size(); ++i) {
        if (nullptr == cmdBuffer.commands[i].op.get()) {
            continue;
        }
        if (cmdBuffer.commands[i].op->opcode_index == reshapeCode) {
            reshapeIndexes.emplace_back(i);
            if (TensorUtils::getDescribe(cmdBuffer.commands[i].outputs[0])->usage != Tensor::InsideDescribe::OUTPUT) {
                reshapeOutput.insert(std::make_pair(cmdBuffer.commands[i].outputs[0], i));
            }
            continue;
        }
    }
    bool change;
    do {
        change = false;
        for (int i=1; i<reshapeIndexes.size(); ++i) {
            auto& curCmd = cmdBuffer.commands[reshapeIndexes[i]];
            if (curCmd.op.get() == nullptr) {
                continue;
            }
            auto iter = reshapeOutput.find(curCmd.inputs[0]);
            if (iter != reshapeOutput.end()) {
                auto& removeCmd = cmdBuffer.commands[iter->second];
                curCmd.inputs[0] = removeCmd.inputs[0];
                change = true;
                // Change input from
                TensorUtils::getDescribe(iter->first)->useCount--;
                if (TensorUtils::getDescribe(iter->first)->useCount <= 0) {
                    removeCmd.op.reset();
                }
            }
        }
    } while (change);
}
static void _computeRefCount(ConvertTflite::CommandBuffer& cmdBuffer) {
    // Compute RefCount
    for (int i=0; i<cmdBuffer.commands.size(); ++i) {
        if (cmdBuffer.commands[i].op.get() == nullptr) {
            continue;
        }
        for (auto t : cmdBuffer.commands[i].inputs) {
            TensorUtils::getDescribe(t)->useCount = 0;
        }
        for (auto t : cmdBuffer.commands[i].outputs) {
            TensorUtils::getDescribe(t)->useCount = 0;
        }
    }
    for (int i=0; i<cmdBuffer.commands.size(); ++i) {
        for (auto t : cmdBuffer.commands[i].inputs) {
            TensorUtils::getDescribe(t)->useCount++;
        }
    }
}
ConvertTflite::CommandBuffer OptimizeCommandBuffer::reduce(ConvertTflite::CommandBuffer&& cmdBuffer) {
//    _computeRefCount(cmdBuffer);
//    _swapDequantAndReshape(cmdBuffer, mRoot);
    _computeRefCount(cmdBuffer);
    _removeDupReshape(cmdBuffer, mRoot);

    return std::move(cmdBuffer);
}
};
