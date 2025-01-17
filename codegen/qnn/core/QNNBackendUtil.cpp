#include "QNNBackend.hpp"

namespace MNN {
// QNNRuntime
ErrorCode QNNRuntime::initDevice() {
    // set the engine interface
    auto statusCode = qnn::tools::dynamicloadutil::getQnnFunctionPointers("libQnnHtp.so",
                                                                          &m_qnnFunctionPointers,
                                                                          &m_backendLibraryHandle,
                                                                          nullptr);
    // Because we build graph in runtime, the freeGraphInfoFnHandle should be assigned here
    m_qnnFunctionPointers.freeGraphInfoFnHandle = QNNBackend::QnnModel_freeGraphsInfo;
    auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
        m_logHandle, (const QnnBackend_Config_t **)m_backendConfig, &m_backendHandle);
    if (qnnStatus != QNN_BACKEND_NO_ERROR) {
        MNN_ERROR("Could not initialize QNNRuntime due to Error %d\n", (int)qnnStatus);
    }
    return NO_ERROR;
}
ErrorCode QNNRuntime::setupResourcePool() {
    return NO_ERROR;
}
void QNNRuntime::releaseResourcePool(int level) {

}
bool QNNRuntime::onSetCache(const void* buffer, size_t size) {
    return true;
}
std::pair<const void*, size_t> QNNRuntime::onGetCache() {
    return std::make_pair(nullptr, 0);
}


// QNNBackend
void QNNBackend::initParam() {

}
ErrorCode QNNBackend::jitPreBuild() {
    return NO_ERROR;
}
ErrorCode QNNBackend::jitPreTuning() {
    return NO_ERROR;
}


}