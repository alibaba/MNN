#ifndef OPCOUNT_HPP
#define OPCOUNT_HPP

#include <MNN/MNNDefine.h>
#include <string>
#include <map>
#include <set>
namespace MNN {
class MNN_PUBLIC OpCount {
public:
    OpCount();
    ~OpCount();
    static OpCount* get();
    void insertOp(const std::string& framework, const std::string& name);
    const std::map<std::string, std::set<std::string>>& getMap();
private:
    std::map<std::string, std::set<std::string>> mOps;
};
};


#endif
