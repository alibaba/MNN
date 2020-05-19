//
//  TFliteExtraManager.hpp
//  MNNConverter
//
//  Created by MNN on 2020/03/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include <MNN/expr/ExprCreator.hpp>
namespace MNN {
namespace Express {
class TFliteExtraManager {
public:
    class Transform {
    public:
        virtual ~ Transform() = default;
        Transform() = default;

        virtual EXPRP onExecute(EXPRP expr) const = 0;
    };

    void insert(const std::string& name, std::shared_ptr<Transform> transform);
    std::shared_ptr<Transform> find(const std::string& name) const;
    static std::shared_ptr<TFliteExtraManager> get();
private:
    std::map<std::string, std::shared_ptr<Transform>> mTransform;
    static std::shared_ptr<TFliteExtraManager> gInstance;
};
}
}
