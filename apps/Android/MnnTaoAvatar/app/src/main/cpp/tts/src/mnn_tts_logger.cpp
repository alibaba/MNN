#include "mnn_tts_logger.hpp"

MNNITTSLogger &MNNITTSLogger::GetInstance()
{
    static MNNITTSLogger instance;
    return instance;
}

void MNNITTSLogger::SetLogLevel(PixelAITTSLogLevel level)
{
    log_level_ = level;
}

std::string MNNITTSLogger::LevelToString(PixelAITTSLogLevel level)
{
    switch (level)
    {
    case TRACE:
        return "TRACE";
    case PDEBUG:
        return "PDEBUG";
    case INFO:
        return "INFO";
    case WARNING:
        return "WARNING";
    case ERROR:
        return "ERROR";
    case CRITICAL:
        return "CRITICAL";
    default:
        return "UNKNOWN";
    }
}
