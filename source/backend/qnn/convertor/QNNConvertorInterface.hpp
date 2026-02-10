#ifndef MNN_QNNCONVERTORINTERFACE_HPP
#define MNN_QNNCONVERTORINTERFACE_HPP

#include "QNNUtils.hpp"

namespace MNN {
namespace QNN {

#ifdef ENABLE_QNN_ONLINE_FINALIZE

extern QNN_INTERFACE_VER_TYPE gQnnConvertorInterface;
// Add gQnnConvertorSystemInterface only for passing compilation check when the Convert mode is on.
extern QNN_SYSTEM_INTERFACE_VER_TYPE gQnnConvertorSystemInterface;
#endif
} // end namespace MNN
} // end namespace QNN
#endif // end MNN_QNNCONVERTORINTERFACE_HPP
