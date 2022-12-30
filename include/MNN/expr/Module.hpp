//
//  Module.hpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Train_Module_hpp
#define MNN_Train_Module_hpp

#include <vector>
#include <unordered_map>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/MNNForwardType.h>

namespace MNN {
namespace Express {
struct SubGraph;
class MNN_PUBLIC Module {
public:
    Module()                                                                               = default;
    virtual ~Module()                                                                      = default;
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) = 0;
    Express::VARP forward(Express::VARP input);
    std::vector<Express::VARP> parameters() const;
    bool loadParameters(const std::vector<Express::VARP>& parameters);
    void setIsTraining(const bool isTraining);
    bool getIsTraining();
    void clearCache();
    const std::string& name() const {
        return mName;
    };
    void setName(std::string name) {
        mName = std::move(name);
    }
    const std::string type() const {
        return mType;
    }
    void setType(std::string type) {
        mType = std::move(type);
    }
    // Return the parameter index
    int addParameter(Express::VARP parameter);

    void setParameter(Express::VARP parameter, int index);
    static Module* createEmpty(const std::vector<Express::VARP>& parameters);
    
    struct BackendInfo {
        MNNForwardType type = MNN_FORWARD_CPU;
        BackendConfig* config = nullptr;
    };
    
    struct Config {
        // Load module as dynamic, default static
        bool dynamic = false;

        // for static mode, if the shape is mutable, set true, otherwise set false to avoid resizeSession freqencily
        bool shapeMutable = true;
        // Pre-rearrange weights or not. Disabled by default.
        // The weights will be rearranged in a general way, so the best implementation
        // may not be adopted if `rearrange` is enabled.
        bool rearrange = false;
        
        BackendInfo* backend = nullptr;
    };
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const Config* config = nullptr);
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const Config* config = nullptr);
    // Shared RuntimeManager
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Config* config = nullptr);
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Config* config = nullptr);
    
    static Module* extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph = {});

    static Module* clone(const Module* module, const bool shareParams = false);

    struct Info {
        // Input info load from model
        std::vector<Variable::Info> inputs;
        // The Module's defaultFormat, NCHW or NHWC
        Dimensionformat defaultFormat;
        // Runtime Info
        std::shared_ptr<MNN::Express::Executor::RuntimeManager> runTimeManager;
        // Input Names By Order
        std::vector<std::string> inputNames;
        // Output Names By Order
        std::vector<std::string> outputNames;
        // The MNNConvert's Version build the module
        std::string version;
    };
    const Info* getInfo() const;
    class CloneContext {
    public:
        CloneContext() = default;
        explicit CloneContext(const bool shareParams)
            : mShareParams(shareParams) {}
        virtual ~CloneContext() = default;

        const bool shareParams() const { return mShareParams; }

        EXPRP getOrClone(const EXPRP expr);
        VARP getOrClone(const VARP var);

    private:
        bool mShareParams = false;
        std::unordered_map<const Expr*, EXPRP> mExprMap;
        std::unordered_map<const Variable*, VARP> mVarMap;
    };

    virtual Module* clone(CloneContext* ctx) const {
        return nullptr;
    }
    void registerModel(const std::vector<std::shared_ptr<Module>>& children);

    static void destroy(Module* m);
protected:
    virtual void onClearCache() {
    }

    Module* cloneBaseTo(CloneContext* ctx, Module* module) const;

private:
    void _collectParameters(std::vector<Express::VARP>& result) const;
    std::vector<std::shared_ptr<Module>> mChildren;
    std::vector<Express::VARP> mParameters;
    bool mIsTraining = true;
    std::string mName;
    std::string mType;
};

struct SubGraph {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::shared_ptr<Module> m;
};

} // namespace Train
} // namespace MNN

#endif
