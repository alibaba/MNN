//
//  Module.hpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Train_Module_hpp
#define MNN_Train_Module_hpp
#include <MNN/expr/Expr.hpp>
#include <set>
namespace MNN {
namespace Train {
class MNN_PUBLIC Module {
public:
    Module()                                                                               = default;
    virtual ~Module()                                                                      = default;
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) = 0;
    Express::VARP forward(Express::VARP input);
    std::set<Express::VARP> parameters() const;
    void setIsTraining(const bool isTraining);
    bool getIsTraining();
    static std::shared_ptr<Module> transform(const std::vector<Express::VARP>& inputs,
                                             const std::vector<Express::VARP>& outputs);

    void clearCache();

protected:
    void registerModel(const std::vector<std::shared_ptr<Module>>& children);
    void addParameter(Express::VARP parameter);
    virtual void onClearCache() {
    }

private:
    void _collectParameters(std::set<Express::VARP>& result) const;
    std::vector<std::shared_ptr<Module>> mChildren;
    std::vector<Express::VARP> mParameters;
    bool mIsTraining = true;
};
} // namespace Train
} // namespace MNN

#endif
