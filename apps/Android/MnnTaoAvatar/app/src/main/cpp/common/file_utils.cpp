#include "file_utils.hpp"
#include "mh_log.hpp"

std::vector<char> file_utils::LoadFileToBuffer(const char *fileName)
{
    if (!fileName) {
        MH_ERROR("Load file error: Invalid file name.");
        return {};
    }

    file_loader fileLoader(fileName);
    auto rawBuffer = fileLoader.read();
    std::vector<char> fileContent(reinterpret_cast<char *>(rawBuffer->buffer()), reinterpret_cast<char *>(rawBuffer->buffer()) + rawBuffer->length());

    return fileContent;
}