//
//  CaffeExtraManager.hpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include <MNN/expr/ExprCreator.hpp>
namespace MNN {
namespace Express {
class CaffeExtraManager {
public:
    class Transform {
    public:
        virtual ~ Transform() = default;
        Transform() = default;
        
        virtual EXPRP onExecute(EXPRP expr) const = 0;
    };
    
    void insert(const std::string& name, std::shared_ptr<Transform> transform);
    std::shared_ptr<Transform> find(const std::string& name) const;
    static CaffeExtraManager *get();
private:
    std::map<std::string, std::shared_ptr<Transform>> mTransform;
};
}  // namespace Express
} // namespace MNN
