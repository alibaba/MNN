//
//  NPUDepthToSpace.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUDepthToSpace.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUDepthToSpace::NPUDepthToSpace(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    
}

ErrorCode NPUDepthToSpace::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::DepthToSpace> depthToSpace(new hiai::op::DepthToSpace(opName));
    shared_ptr<ge::op::Permute> permuteBefore(new ge::op::Permute(opName+"_before"));
    shared_ptr<ge::op::Permute> permuteAfter(new ge::op::Permute(opName+"_after"));

    /*
     * set om op
     * */

    // 
    auto inputIndex1 = mOp->inputIndexes()->data()[0];
    auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
    auto xOp1        = iops1.back().first;

    auto param = mOp->main_as_DepthSpaceParam();

    (*permuteBefore)
        .set_input_x(*xOp1.get())
        .set_attr_order({0,2,3,1})
        .SetAttr("NCHW_to_NHWC", ge::AttrValue::CreateFrom<ge::AttrValue::INT>(1));
    
    (*depthToSpace)
        .set_input_x(*permuteBefore.get())
        .set_attr_block_size(ge::AttrValue::INT(param->blockSize()))
        .set_attr_data_format("NHWC");

    (*permuteAfter)
        .set_input_x(*depthToSpace.get())
        .set_attr_order({0,3,1,2})
        .SetAttr("NHWC_to_NCHW", ge::AttrValue::CreateFrom<ge::AttrValue::INT>(1));

    mNpuBackend->setOutputOps(mOp, {permuteBefore, depthToSpace, permuteAfter}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUDepthToSpace>> __depth_to_space_op(OpType_DepthToSpace);

} // namespace MNN