#include "a2bs/mnn_audio2bs_logger.hpp"

PixelAIAudio2BSLogger &PixelAIAudio2BSLogger::GetInstance()
{
    static PixelAIAudio2BSLogger instance;
    return instance;
}

std::string PixelAIAudio2BSLogger::LevelToString(PixelAIAudio2BSLogLevel level)
{
    switch (level)
    {
    case ATB_TRACE:
        return "TRACE";
    case ATB_DEBUG:
        return "PDEBUG";
    case ATB_INFO:
        return "INFO";
    case ATB_WARNING:
        return "WARNING";
    case ATB_ERROR:
        return "ERROR";
    case ATB_CRITICAL:
        return "CRITICAL";
    default:
        return "UNKNOWN";
    }
}
