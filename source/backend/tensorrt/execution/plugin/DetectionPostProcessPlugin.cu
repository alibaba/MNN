#include "DetectionPostProcessPlugin.hpp"

namespace MNN {

    template <typename T>
    __global__ void decodeBoxes_kernel(const int count, const float* boxesPtr, const CenterSizeEncoding* anchorsPtr, BoxCornerEncoding* decodeBoxesPtr, const CenterSizeEncoding& scaleValues, int numBoxes, int boxCoordNum, int anchorsCoordNum, int numAnchors1) {
        CUDA_KERNEL_LOOP(idx, count) {
            const int boxIndex = idx * boxCoordNum;
            CenterSizeEncoding boxCenterSize      = *reinterpret_cast<const CenterSizeEncoding*>(boxesPtr + boxIndex);
            CenterSizeEncoding anchor             = anchorsPtr[idx];
            float ycenter      = boxCenterSize.y / scaleValues.y * anchor.h + anchor.y;
            float xcenter      = boxCenterSize.x / scaleValues.x * anchor.w + anchor.x;
            float halfh        = 0.5f * static_cast<float>(exp(boxCenterSize.h / scaleValues.h)) * anchor.h;
            float halfw        = 0.5f * static_cast<float>(exp(boxCenterSize.w / scaleValues.w)) * anchor.w;
            auto& curBox       = decodeBoxesPtr[idx];
            curBox.ymin        = ycenter - halfh;
            curBox.xmin        = xcenter - halfw;
            curBox.ymax        = ycenter + halfh;
            curBox.xmax        = xcenter + halfw;
        }
    }


    template <typename T>
    __global__ void maxScores_kernel(const int count, const float* scoresStartPtr, int numClassWithBackground, int labelOffset, int* sortedClassIndicesPtr, int numClasses, int numCategoriesPerAnchor, float* maxScores){
        CUDA_KERNEL_LOOP(idx, count) {
            const auto boxScores = scoresStartPtr + idx * numClassWithBackground + labelOffset;
            int* classIndices    = sortedClassIndicesPtr + idx * numClasses;

            // iota(classIndices, numClasses, 0);
            int data = 0;
            for(int i = 0; i < numClasses; i++){
                classIndices[i] = data;
                data += 1;
            }
            // std::partial_sort(classIndices, classIndices + numCategoriesPerAnchor, classIndices + numClasses,
            //                 [&boxScores](const int i, const int j) { return boxScores[i] > boxScores[j]; });
            
            int score = classIndices[0];
            for(int i = 0; i < numClasses; i++){
                score = max(classIndices[i], score);
            }
            maxScores[idx] = boxScores[score];
        }
    }


    template <typename T>
    __global__ void copy_candidate(const int count, Candidate* candidatePtr, const float* score){
        CUDA_KERNEL_LOOP(idx, count) {
            int index = 0;
            for(int i = 0; i < count; i++){
                if(score[idx] < score[i]){
                    index++;
                }
            }
            candidatePtr[idx].index = index;
            candidatePtr[idx].boxIndex = idx;
            candidatePtr[idx].score = score[idx];
        }
    }

    __device__ __forceinline__ float IOU(const float* boxes, int i, int j) {
        const float yMinI = min(boxes[i * 4 + 0], boxes[i * 4 + 2]);
        const float xMinI = min(boxes[i * 4 + 1], boxes[i * 4 + 3]);
        const float yMaxI = max(boxes[i * 4 + 0], boxes[i * 4 + 2]);
        const float xMaxI = max(boxes[i * 4 + 1], boxes[i * 4 + 3]);
        const float yMinJ = min(boxes[j * 4 + 0], boxes[j * 4 + 2]);
        const float xMinJ = min(boxes[j * 4 + 1], boxes[j * 4 + 3]);
        const float yMaxJ = max(boxes[j * 4 + 0], boxes[j * 4 + 2]);
        const float xMaxJ = max(boxes[j * 4 + 1], boxes[j * 4 + 3]);
        const float areaI = (yMaxI - yMinI) * (xMaxI - xMinI);
        const float areaJ = (yMaxJ - yMinJ) * (xMaxJ - xMinJ);
        if (areaI <= 0 || areaJ <= 0)
            return 0.0;
        const float intersectionYMin = max(yMinI, yMinJ);
        const float intersectionXMin = max(xMinI, xMinJ);
        const float intersectionYMax = min(yMaxI, yMaxJ);
        const float intersectionXMax = min(xMaxI, xMaxJ);
        const float intersectionArea = max(intersectionYMax - intersectionYMin, 0.0) *
                                       max(intersectionXMax - intersectionXMin, 0.0);
        return intersectionArea / (areaI + areaJ - intersectionArea);
    }

    template <typename T>
    __global__ void nms_kernel(const int count, int numBoxes, float scoreThreshold, float iouThreshold, Candidate* candidatePtr, int* selectedSize, float* decodedBoxesPtr, int* selectedPtr){
        CUDA_KERNEL_LOOP(idx, count) {
            int boxIndex = 0;
            float originalScore = 0; 
            for(int i = 0; i < numBoxes; i++){
                if(candidatePtr[i].index == idx){
                    boxIndex = candidatePtr[i].boxIndex;
                    originalScore = candidatePtr[i].score;
                }
            }

            if(originalScore <= scoreThreshold){
                return;
            }
            
            bool shouldSelect = true;

            for (int j = (selectedSize[0] - 1); j >= 0; --j) {
                float iou = IOU(decodedBoxesPtr, boxIndex, selectedPtr[j]);
                if (iou == 0.0) {
                    continue;
                }
                if (iou > iouThreshold) {
                    shouldSelect = false;
                }
            }

            if (shouldSelect) {
                selectedPtr[selectedSize[0]] = boxIndex;
                atomicAdd(selectedSize, 1);
            }

        }
    }

    template <typename T>
    __global__ void set_output(const int count, const BoxCornerEncoding* decodedBoxesPtr, BoxCornerEncoding* detectionBoxesPtr, float* detectionClassesPtr, float* detectionScoresPtr, float* numDetectionsPtr, const float* scoresStartPtr, int numClassWithBackground, int labelOffset, int* sortedClassIndicesPtr, int numClasses, int numCategoriesPerAnchor, int* selectedPtr){
        CUDA_KERNEL_LOOP(index, count) {
            int selectedIndex = selectedPtr[index];
            const float* boxScores  = scoresStartPtr + selectedIndex * numClassWithBackground + labelOffset;
            const int* classIndices = sortedClassIndicesPtr + selectedIndex * numClasses;
            for (int col = 0; col < numCategoriesPerAnchor; ++col) {
                int boxOffset                  = numCategoriesPerAnchor * numDetectionsPtr[0] + col;
                detectionBoxesPtr[boxOffset]   = decodedBoxesPtr[selectedIndex];
                detectionClassesPtr[boxOffset] = classIndices[col];
                detectionScoresPtr[boxOffset]  = boxScores[classIndices[col]];
                atomicAdd(numDetectionsPtr, 1);
            }
        }
    }


    void DetectionPostProcessPlugin::decodeBoxes(nvinfer1::DataType dataType, const int count, const void *const * inputs, const void *const * outputs, const void * scaleValues, void * decodeBoxes, int numBoxes, int boxCoordNum, int anchorsCoordNum, int numAnchors1) {
        auto boxesEncoding = inputs[0];
        auto anchors = inputs[2];
        if (dataType == nvinfer1::DataType::kFLOAT){
            return decodeBoxes_kernel<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, (float*)boxesEncoding, reinterpret_cast<const CenterSizeEncoding*>(anchors), reinterpret_cast<BoxCornerEncoding*>(decodeBoxes), *reinterpret_cast<const CenterSizeEncoding*>(scaleValues), numBoxes, boxCoordNum, anchorsCoordNum, numAnchors1);
        }else{
            return decodeBoxes_kernel<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, (float*)boxesEncoding, reinterpret_cast<const CenterSizeEncoding*>(anchors), reinterpret_cast<BoxCornerEncoding*>(decodeBoxes), *reinterpret_cast<const CenterSizeEncoding*>(scaleValues), numBoxes, boxCoordNum, anchorsCoordNum, numAnchors1);
        }
    }

    void DetectionPostProcessPlugin::maxScores(nvinfer1::DataType dataType, const int count, const void *const * inputs, const void *const * outputs, int numClassWithBackground, int* sortedClassIndicesPtr, int numClasses, float* maxScores, int maxClassesPerAnchor) {
        auto classPredictions = inputs[1];
        const int labelOffset = numClassWithBackground - numClasses;
        int numCategoriesPerAnchor = std::min(maxClassesPerAnchor, numClasses);
        maxScores_kernel<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, (const float*)classPredictions, numClassWithBackground, labelOffset, sortedClassIndicesPtr, numClasses, numCategoriesPerAnchor, maxScores);
    }

    void DetectionPostProcessPlugin::NMSSingleClasss(float* decodedBoxesPtr, const float* scoresPtr, int maxDetections,
        float iouThreshold, float scoreThreshold, int* selectedPtr, int* selectedSize, int numBoxes, int outputNum, Candidate* candidate, Candidate* mCandidatePriorityQueue){
            copy_candidate<float><<<CAFFE_GET_BLOCKS(numBoxes), CUDA_NUM_THREADS>>>(numBoxes, candidate, scoresPtr);
            nms_kernel<float><<<CAFFE_GET_BLOCKS(outputNum), CUDA_NUM_THREADS>>>(outputNum, numBoxes, scoreThreshold, iouThreshold, candidate, selectedSize, decodedBoxesPtr, selectedPtr);
    }

    void DetectionPostProcessPlugin::setOutput(const int selectSize, const BoxCornerEncoding* decodedBoxesPtr, BoxCornerEncoding* detectionBoxesPtr, float* detectionClassesPtr, float* detectionScoresPtr, float* numDetectionsPtr, const float* scoresStartPtr, int numClassWithBackground, int labelOffset, int* sortedClassIndicesPtr, int numClasses, int numCategoriesPerAnchor, int* selectedPtr){
        set_output<float><<<CAFFE_GET_BLOCKS(selectSize), CUDA_NUM_THREADS>>>(selectSize, decodedBoxesPtr, detectionBoxesPtr, detectionClassesPtr, detectionScoresPtr, numDetectionsPtr, scoresStartPtr, numClassWithBackground, labelOffset, sortedClassIndicesPtr, numClasses, numCategoriesPerAnchor, selectedPtr);
    }

}; // namespace MNN