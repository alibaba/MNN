//
//  NN.hpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Train_NN_hpp
#define MNN_Train_NN_hpp
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <vector>
namespace MNN {
namespace Express {
class Initializer;

class MNN_PUBLIC NN {
public:
    enum ActivationFunctionType {
        None = 0,
        Relu = 1,
        Relu6 = 2,
    };
    enum ScaleUpdateMethod {
        Maximum = 0,
        MovingAverage = 1
    };
    enum FeatureScaleStatMethod {
        PerTensor = 0,
        PerChannel = 1 // Depercerate
    };
    /* Unlike enum in class, class in class need be dllimport or dllexport explcility.
       Compiling in other system will not be affected.
     */
    struct MNN_PUBLIC ConvOption {
        Express::INTS kernelSize     = {1, 1};
        Express::INTS channel        = {0, 0};
        Express::INTS stride         = {1, 1};
        Express::INTS dilate         = {1, 1};
        Express::PaddingMode padMode = Express::VALID;
        Express::INTS pads           = {0, 0};
        bool depthwise               = false;
        ActivationFunctionType fusedActivationFunction = None;
        void reset(int size = 2);
    };
    static Module* Conv(const ConvOption& option, bool bias = true,
                                        std::shared_ptr<Initializer> weightInit = nullptr,
                                        std::shared_ptr<Initializer> biasInit   = nullptr);
    static Module* ConvTranspose(const ConvOption& option, bool bias = true,
                                                 std::shared_ptr<Initializer> weightInit = nullptr,
                                                 std::shared_ptr<Initializer> biasInit   = nullptr);
    static Module* Linear(int l, int t, bool hasBias = true,
                                          std::shared_ptr<Initializer> weightInit = nullptr,
                                          std::shared_ptr<Initializer> biasInit   = nullptr);
    static Module* Dropout(const float dropRatio);
    static Module* BatchNorm(const int channels, const int dims = 4, const float m = 0.999,
                                             const float e = 1e-5);

    static Module* ConvInt8(const ConvOption& option, int bits = 8, bool bias = true,
                                            std::shared_ptr<Initializer> weightInit = nullptr,
                                            std::shared_ptr<Initializer> biasInit   = nullptr,
                                            FeatureScaleStatMethod featureMethod = PerChannel,
                                            ScaleUpdateMethod method = MovingAverage
                                            );
    struct ConvParameters {
        ConvOption option;
        Express::VARP weight;
        Express::VARP bias;
        int group;
        std::string name;
    };
    static Module* ConvInt8(const ConvParameters& parameters, int bits,
                                            FeatureScaleStatMethod featureMethod = PerChannel,
                                            ScaleUpdateMethod method = MovingAverage);
    static Module* Conv(const ConvParameters& parameters);
    static Module* ConvBNReluFused(std::vector<std::shared_ptr<Module> > modules,
                                                   NN::FeatureScaleStatMethod featureScaleStatMethod = PerTensor,
                                                   NN::ScaleUpdateMethod scaleUpdateMethod = MovingAverage, const int bits = 8, bool winograd = false);

    class Utils {
    public:
        // ConvOption, Weight, Bias, Group
        static ConvParameters ExtractConvolution(Express::EXPRP expr);

        // Extract BatchNormal and Dropout
        static Module* ExtractNotRunableOp(Express::EXPRP expr, const std::map<std::string, SubGraph>& subgraphs);
    };
    
    static bool turnQuantize(Module* module, const int bits = 8, NN::FeatureScaleStatMethod featureScaleStatMethod = NN::PerTensor, NN::ScaleUpdateMethod scaleUpdateMethod = NN::MovingAverage, bool winogradOpt = false);
    static Module* extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph = {});
};

} // namespace Train
} // namespace MNN

#endif
