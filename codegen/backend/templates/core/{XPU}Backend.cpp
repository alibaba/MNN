{COPYRIGHT}


{EXTRA_INCLUDE_FILES}
// MNN headers
#include "{XPU}Backend.hpp"
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {

// Runtime Part
// <begin Constructor and Destructor
{XPU}Runtime::{XPU}Runtime(const Backend::Info& info) {
    // 0. initialize all parameters
    mInfo = info;
    mPower     = BackendConfig::Power_Normal;
    mMemory    = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
    }
    // 1. Device & Library setup
    auto err = initDevice();
    if (err != ErrorCode::NO_ERROR) {
        MNN_ERROR("[{XPU}]: initDevice failed!\n");
    }
    // 2. Resource (Buffer) Pool setup
    err = setupResourcePool();
    if (err != ErrorCode::NO_ERROR) {
        MNN_ERROR("[{XPU}]: setupResourcePool failed!\n");
    }
}
{XPU}Runtime::~{XPU}Runtime() {
    // release ResourcePool
    releaseResourcePool();
}

Backend* {XPU}Runtime::onCreate(const BackendConfig* config, Backend* origin) const {
    if (config != nullptr) {
        mPrecision = config->precision;
        mPower     = config->power;
        mMemory    = config->memory;
    }
    return new {XPU}Backend(this);
}
Runtime::CompilerType {XPU}Runtime::onGetCompilerType() const {
    return Compiler_Origin;
}
void {XPU}Runtime::onGabageCollect(int level) {
    // release ResourcePool
    releaseResourcePool(level);
}

// Backend Part
// <begin Constructor and Destructor
{XPU}Backend::{XPU}Backend(MNNForwardType type, {XPU}Runtime* rt) : Backend(type) {
    // 0. initialize all parameters
    mRuntime = rt;
    mPower     = mRuntime->mPower;
    mMemory    = mRuntime->mMemory;
    mPrecision = mRuntime->mPrecision;
    initParam();
    // 1. possible jit pre-build & pre-tuning setup
    jitPreBuild();
    jitPreTuning();
}
{XPU}Backend::~{XPU}Backend() {
    // release all temporary resources
}
// end Constructor and Destructor>

// <begin Execution Registration & Creation
// static OpType -> Execution Creator map
static inline std::map<OpType, {XPU}Backend::Creator*>* getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, {XPU}Backend::Creator*>* ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, {XPU}Backend::Creator*>; });
    return ret;
}
bool {XPU}Backend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be multi-added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}
Execution* {XPU}Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {

    auto map = getCreatorMap();
    auto iter = map->find(op->type());
    
    // error handle 1
    if (iter == map->end()) {
        MNN_ERROR("map not find !!! \n");
        if(op != nullptr){
            if(op->name() != nullptr){
                MNN_PRINT("[{XPU}] Don't support type %d, %s\n", op->type(), op->name()->c_str());
            }
        }
        return nullptr;
    }

    // Create
    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    // error handle 2
    if (nullptr == exe) {
        MNN_ERROR("nullptr == exe !!! \n");
        if(op != nullptr){
            if(op->name() != nullptr){
                MNN_PRINT("[{XPU}] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
            }
        }
        return nullptr;
    }

    return exe;
}
// end Execution Registration & Creation>

// 2. Pipeline Functions
void {XPU}Backend::onExecuteBegin() const {

}
void {XPU}Backend::onExecuteEnd() const {

}
void {XPU}Backend::onResizeBegin() {

}
ErrorCode {XPU}Backend::onResizeEnd() {
    return NO_ERROR;
}

// 3. Buffer Management
MemObj* {XPU}Backend::onAcquire(const Tensor* tensor, StorageType storageType) {
    return nullptr;
}
bool {XPU}Backend::onClearBuffer() {
    return true;
}
void {XPU}Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {

}


// <begin insert new runtime to creator
class {XPU}RuntimeCreator : public RuntimeCreator {
    Runtime* onCreate(const Backend::Info &info) const {
        return std::static_cast<Runtime*>(new {XPU}Runtime(info));
    }
    bool onValid(Backend::Info& info) const {
        return true;
    }
};

static const auto __{XPU_LOWER}_global_initializer = []() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_{XPU_FOWARD_TYPE}, new {XPU}RuntimeCreator, true);
    return true;
}();
// insertion end>
}