//
//  multiPose.cpp
//  MNN
//
//  Created by MNN on 2018/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include "PoseNames.hpp"

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN;

#define MODEL_IMAGE_SIZE 513
#define OUTPUT_STRIDE 16

#define MAX_POSE_DETECTIONS 10
#define NUM_KEYPOINTS 17
#define SCORE_THRESHOLD 0.5
#define MIN_POSE_SCORE 0.25
#define NMS_RADIUS 20
#define LOCAL_MAXIMUM_RADIUS 1

#define OFFSET_NODE_NAME "offset_2"
#define DISPLACE_FWD_NODE_NAME "displacement_fwd_2"
#define DISPLACE_BWD_NODE_NAME "displacement_bwd_2"
#define HEATMAPS "heatmap"

#define CIRCLE_RADIUS 3

inline float clip(float value, float min, float max) {
    if (value < 0) {
        return 0;
    } else if (value > max) {
        return max;
    } else {
        return value;
    }
}

static int changeColorCircle(uint32_t* src, CV::Point point, int width, int height) {
    for (int y = -CIRCLE_RADIUS; y < (CIRCLE_RADIUS + 1); ++y) {
        for (int x = -CIRCLE_RADIUS; x < (CIRCLE_RADIUS + 1); ++x) {
            const int xx = static_cast<int>(point.fX + x);
            const int yy = static_cast<int>(point.fY + y);
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                int index  = yy * width + xx;
                src[index] = 0xFFFF00FF;
            }
        }
    }

    return 0;
}

static int drawPose(uint8_t* rgbaPtr, int width, int height, std::vector<float>& poseScores,
                    std::vector<std::vector<float>>& poseKeypointScores,
                    std::vector<std::vector<CV::Point>>& poseKeypointCoords) {
    const int poseCount = poseScores.size();
    for (int i = 0; i < poseCount; ++i) {
        if (poseScores[i] > MIN_POSE_SCORE) {
            for (int id = 0; id < NUM_KEYPOINTS; ++id) {
                if (poseKeypointScores[i][id] > SCORE_THRESHOLD) {
                    CV::Point point = poseKeypointCoords[i][id];
                    changeColorCircle((uint32_t*)rgbaPtr, point, width, height);
                }
            }
        }
    }

    return 0;
}

static CV::Point getCoordsFromTensor(const Tensor* dataTensor, int id, int x, int y, bool getCoord = true) {
    // dataTensor must be [1,c,h,w]
    auto dataPtr         = dataTensor->host<float>();
    const int xOffset    = dataTensor->channel() / 2;
    const int indexPlane = y * dataTensor->stride(2) + x;
    const int indexY     = id * dataTensor->stride(1) + indexPlane;
    const int indexX     = (id + xOffset) * dataTensor->stride(1) + indexPlane;
    CV::Point point;
    if (getCoord) {
        point.set(dataPtr[indexX], dataPtr[indexY]);
    } else {
        point.set(0.0, dataPtr[indexY]);
    }
    return point;
};

// decode pose and posenet model reference from https://github.com/rwightman/posenet-python
static int decodePoseImpl(float curScore, int curId, const CV::Point& originalOnImageCoords, const Tensor* heatmaps,
                          const Tensor* offsets, const Tensor* displacementFwd, const Tensor* displacementBwd,
                          std::vector<float>& instanceKeypointScores, std::vector<CV::Point>& instanceKeypointCoords) {
    instanceKeypointScores[curId] = curScore;
    instanceKeypointCoords[curId] = originalOnImageCoords;
    const int height              = heatmaps->height();
    const int width               = heatmaps->width();
    std::map<std::string, int> poseNamesID;
    for (int i = 0; i < PoseNames.size(); ++i) {
        poseNamesID[PoseNames[i]] = i;
    }

    auto traverseToTargetKeypoint = [=](int edgeId, const CV::Point& sourcekeypointCoord, int targetKeypointId,
                                        const Tensor* displacement) {
        int sourceKeypointIndicesX =
            static_cast<int>(clip(round(sourcekeypointCoord.fX / (float)OUTPUT_STRIDE), 0, (float)(width - 1)));
        int sourceKeypointIndicesY =
            static_cast<int>(clip(round(sourcekeypointCoord.fY / (float)OUTPUT_STRIDE), 0, (float)(height - 1)));

        auto displacementCoord =
            getCoordsFromTensor(displacement, edgeId, sourceKeypointIndicesX, sourceKeypointIndicesY);
        float displacedPointX = sourcekeypointCoord.fX + displacementCoord.fX;
        float displacedPointY = sourcekeypointCoord.fY + displacementCoord.fY;

        int displacedPointIndicesX =
            static_cast<int>(clip(round(displacedPointX / OUTPUT_STRIDE), 0, (float)(width - 1)));
        int displacedPointIndicesY =
            static_cast<int>(clip(round(displacedPointY / OUTPUT_STRIDE), 0, (float)(height - 1)));

        float score =
            getCoordsFromTensor(heatmaps, targetKeypointId, displacedPointIndicesX, displacedPointIndicesY, false).fY;
        auto offset = getCoordsFromTensor(offsets, targetKeypointId, displacedPointIndicesX, displacedPointIndicesY);

        CV::Point imageCoord;
        imageCoord.fX = displacedPointIndicesX * OUTPUT_STRIDE + offset.fX;
        imageCoord.fY = displacedPointIndicesY * OUTPUT_STRIDE + offset.fY;

        return std::make_pair(score, imageCoord);
    };

    MNN_ASSERT((NUM_KEYPOINTS - 1) == PoseChain.size());

    for (int edge = PoseChain.size() - 1; edge >= 0; --edge) {
        const int targetKeypointID = poseNamesID[PoseChain[edge].first];
        const int sourceKeypointID = poseNamesID[PoseChain[edge].second];
        if (instanceKeypointScores[sourceKeypointID] > 0.0 && instanceKeypointScores[targetKeypointID] == 0.0) {
            auto curInstance = traverseToTargetKeypoint(edge, instanceKeypointCoords[sourceKeypointID],
                                                        targetKeypointID, displacementBwd);
            instanceKeypointScores[targetKeypointID] = curInstance.first;
            instanceKeypointCoords[targetKeypointID] = curInstance.second;
        }
    }

    for (int edge = 0; edge < PoseChain.size(); ++edge) {
        const int sourceKeypointID = poseNamesID[PoseChain[edge].first];
        const int targetKeypointID = poseNamesID[PoseChain[edge].second];
        if (instanceKeypointScores[sourceKeypointID] > 0.0 && instanceKeypointScores[targetKeypointID] == 0.0) {
            auto curInstance = traverseToTargetKeypoint(edge, instanceKeypointCoords[sourceKeypointID],
                                                        targetKeypointID, displacementFwd);
            instanceKeypointScores[targetKeypointID] = curInstance.first;
            instanceKeypointCoords[targetKeypointID] = curInstance.second;
        }
    }

    return 0;
}

static int decodeMultiPose(const Tensor* offsets, const Tensor* displacementFwd, const Tensor* displacementBwd,
                           const Tensor* heatmaps, std::vector<float>& poseScores,
                           std::vector<std::vector<float>>& poseKeypointScores,
                           std::vector<std::vector<CV::Point>>& poseKeypointCoords, CV::Point& scale) {
    // keypoint_id, score, coord((x,y))
    typedef std::pair<int, std::pair<float, CV::Point>> partsType;
    std::vector<partsType> parts;

    const int channel  = heatmaps->channel();
    const int height   = heatmaps->height();
    const int width    = heatmaps->width();
    auto maximumFilter = [&parts, width, height](const int id, const float* startPtr) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // check whether (y,x) is the max value around the neighborhood
                bool isMaxVaule = true;
                float maxValue  = startPtr[y * width + x];
                {
                    for (int i = -LOCAL_MAXIMUM_RADIUS; i < (LOCAL_MAXIMUM_RADIUS + 1); ++i) {
                        for (int j = -LOCAL_MAXIMUM_RADIUS; j < (LOCAL_MAXIMUM_RADIUS + 1); ++j) {
                            float value = 0.0f;
                            int yCoord  = y + i;
                            int xCoord  = x + j;
                            if (yCoord >= 0 && yCoord < height && xCoord >= 0 && xCoord < width) {
                                value = startPtr[yCoord * width + xCoord];
                            }
                            if (maxValue < value) {
                                isMaxVaule = false;
                                break;
                            }
                        }
                    }
                }

                if (isMaxVaule && maxValue >= SCORE_THRESHOLD) {
                    CV::Point coord;
                    coord.set(x, y);
                    parts.push_back(std::make_pair(id, std::make_pair(maxValue, coord)));
                }
            }
        }
    };

    auto scoresPtr = heatmaps->host<float>();

    for (int id = 0; id < channel; ++id) {
        auto idScoresPtr = scoresPtr + id * width * height;
        maximumFilter(id, idScoresPtr);
    }

    // sort the parts according to score
    std::sort(parts.begin(), parts.end(),
              [](const partsType& a, const partsType& b) { return a.second.first > b.second.first; });

    const int squareNMSRadius = NMS_RADIUS * NMS_RADIUS;

    auto withinNMSRadius = [=, &poseKeypointCoords](const CV::Point& point, const int id) {
        bool withinThisPointRadius = false;
        for (int i = 0; i < poseKeypointCoords.size(); ++i) {
            const auto& curPoint = poseKeypointCoords[i][id];
            const auto sum       = powf((curPoint.fX - point.fX), 2) + powf((curPoint.fY - point.fY), 2);
            if (sum <= squareNMSRadius) {
                withinThisPointRadius = true;
                break;
            }
        }
        return withinThisPointRadius;
    };

    std::vector<float> instanceKeypointScores(NUM_KEYPOINTS);
    std::vector<CV::Point> instanceKeypointCoords(NUM_KEYPOINTS);

    auto getInstanceScore = [&]() {
        float notOverlappedScores = 0.0f;
        const int poseNums        = poseKeypointCoords.size();
        if (poseNums == 0) {
            for (int i = 0; i < NUM_KEYPOINTS; ++i) {
                notOverlappedScores += instanceKeypointScores[i];
            }
        } else {
            for (int id = 0; id < NUM_KEYPOINTS; ++id) {
                if (!withinNMSRadius(instanceKeypointCoords[id], id)) {
                    notOverlappedScores += instanceKeypointScores[id];
                }
            }
        }

        return notOverlappedScores / NUM_KEYPOINTS;
    };

    int poseCount = 0;
    for (const auto& part : parts) {
        if (poseCount >= MAX_POSE_DETECTIONS) {
            break;
        }
        const auto curScore  = part.second.first;
        const auto curId     = part.first;
        const auto& curPoint = part.second.second;

        const auto offsetXY = getCoordsFromTensor(offsets, curId, (int)curPoint.fX, (int)curPoint.fY);
        CV::Point originalOnImageCoords;
        originalOnImageCoords.fX = curPoint.fX * OUTPUT_STRIDE + offsetXY.fX;
        originalOnImageCoords.fY = curPoint.fY * OUTPUT_STRIDE + offsetXY.fY;

        if (withinNMSRadius(originalOnImageCoords, curId)) {
            continue;
        }
        ::memset(instanceKeypointScores.data(), 0, sizeof(float) * NUM_KEYPOINTS);
        ::memset(instanceKeypointCoords.data(), 0, sizeof(CV::Point) * NUM_KEYPOINTS);
        decodePoseImpl(curScore, curId, originalOnImageCoords, heatmaps, offsets, displacementFwd, displacementBwd,
                       instanceKeypointScores, instanceKeypointCoords);

        float poseScore = getInstanceScore();
        if (poseScore > MIN_POSE_SCORE) {
            poseScores.push_back(poseScore);
            poseKeypointScores.push_back(instanceKeypointScores);
            poseKeypointCoords.push_back(instanceKeypointCoords);
            poseCount++;
        }
    }

    // scale the pose keypoint coords
    for (int i = 0; i < poseCount; ++i) {
        for (int id = 0; id < NUM_KEYPOINTS; ++id) {
            poseKeypointCoords[i][id].fX *= scale.fX;
            poseKeypointCoords[i][id].fY *= scale.fY;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: ./multiPose.out model.mnn input.jpg pose.jpg" << std::endl;
    }

    const auto poseModel           = argv[1];
    const auto inputImageFileName  = argv[2];
    const auto outputImageFileName = argv[3];

    int originalWidth;
    int originalHeight;
    int originChannel;
    auto inputImage = stbi_load(inputImageFileName, &originalWidth, &originalHeight, &originChannel, 4);
    if (nullptr == inputImage) {
        MNN_ERROR("Invalid path: %s\n", inputImageFileName);
        return 0;
    }

    const int targetWidth  = static_cast<int>((float)originalWidth / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
    const int targetHeight = static_cast<int>((float)originalHeight / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;

    CV::Point scale;
    scale.fX = (float)originalWidth / (float)targetWidth;
    scale.fY = (float)originalHeight / (float)targetHeight;

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    auto session        = mnnNet->createSession(netConfig);

    auto input = mnnNet->getSessionInput(session, nullptr);

    if (input->elementSize() <= 4) {
        mnnNet->resizeTensor(input, {1, 3, targetHeight, targetWidth});
        mnnNet->resizeSession(session);
    }

    // preprocess input image
    {
        const float means[3] = {127.5f, 127.5f, 127.5f};
        const float norms[3] = {2.0f / 255.0f, 2.0f / 255.0f, 2.0f / 255.0f};
        CV::ImageProcess::Config preProcessConfig;
        ::memcpy(preProcessConfig.mean, means, sizeof(means));
        ::memcpy(preProcessConfig.normal, norms, sizeof(norms));
        preProcessConfig.sourceFormat = CV::RGBA;
        preProcessConfig.destFormat   = CV::RGB;
        preProcessConfig.filterType   = CV::BILINEAR;

        auto pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
        CV::Matrix trans;

        // Dst -> [0, 1]
        trans.postScale(1.0 / targetWidth, 1.0 / targetHeight);
        //[0, 1] -> Src
        trans.postScale(originalWidth, originalHeight);

        pretreat->setMatrix(trans);
        const auto rgbaPtr = reinterpret_cast<uint8_t*>(inputImage);
        pretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, input);
    }

    // read image data from txt
    //  {
    //      MNN::Tensor givenTensor(input, Tensor::CAFFE);
    //      std::ifstream inputFile("image.txt");
    //      const int inputSize = givenTensor.elementSize();
    //      auto inputData = givenTensor.host<int8_t>();
    //      int pixel = 0;
    //      for(int i = 0; i < inputSize; ++i){
    //          inputFile >> pixel;
    //          inputData[i] = static_cast<int8_t>(pixel);
    //      }
    //      input->copyFromHostTensor(&givenTensor);
    //  }

    // run...
    {
        AUTOTIME;
        mnnNet->runSession(session);
    }

    // get output
    auto offsets         = mnnNet->getSessionOutput(session, OFFSET_NODE_NAME);
    auto displacementFwd = mnnNet->getSessionOutput(session, DISPLACE_FWD_NODE_NAME);
    auto displacementBwd = mnnNet->getSessionOutput(session, DISPLACE_BWD_NODE_NAME);
    auto heatmaps        = mnnNet->getSessionOutput(session, HEATMAPS);

    Tensor offsetsHost(offsets, Tensor::CAFFE);
    Tensor displacementFwdHost(displacementFwd, Tensor::CAFFE);
    Tensor displacementBwdHost(displacementBwd, Tensor::CAFFE);
    Tensor heatmapsHost(heatmaps, Tensor::CAFFE);

    offsets->copyToHostTensor(&offsetsHost);
    displacementFwd->copyToHostTensor(&displacementFwdHost);
    displacementBwd->copyToHostTensor(&displacementBwdHost);
    heatmaps->copyToHostTensor(&heatmapsHost);

    std::vector<float> poseScores;
    std::vector<std::vector<float>> poseKeypointScores;
    std::vector<std::vector<CV::Point>> poseKeypointCoords;
    {
        AUTOTIME;
        decodeMultiPose(&offsetsHost, &displacementFwdHost, &displacementBwdHost, &heatmapsHost, poseScores,
                        poseKeypointScores, poseKeypointCoords, scale);
    }

    drawPose(inputImage, originalWidth, originalHeight, poseScores, poseKeypointScores, poseKeypointCoords);
    stbi_write_png(outputImageFileName, originalWidth, originalHeight, 4, inputImage, 4 * originalWidth);
    stbi_image_free(inputImage);
    return 0;
}
