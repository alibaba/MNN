//
//  ConvertExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvertExecution_hpp
#define ConvertExecution_hpp

#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"


namespace MNN {
    namespace OpenCL {
        
        class ConvertExecution : public Execution {
        public:
            ConvertExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
            virtual ~ConvertExecution() = default;
            
            virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
            virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
            
        private:
            enum TensorConvertType {
                NHWC2NC4HW4 = 0,
                NC4HW42NHWC = 1,
                NHWC2NCHW   = 2,
                NCHW2NHWC   = 3,
            };
            
            cl::Kernel mKernel;
            uint32_t mMaxWorkGroupSize;
            OpenCLBackend *mOpenCLBackend;
            std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
            std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
        };
        
    } // namespace OpenCL
} // namespace MNN

#endif /* ConvertExecution_hpp */
