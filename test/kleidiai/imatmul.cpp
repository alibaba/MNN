#ifdef MNN_KLEIDIAI_ENABLED

#include <functional>
#include <numeric>
#include <random>

#include "MNNTestSuite.h"
#include "backend/cpu/compute/KleidiAIDenseConvolution.hpp"

using namespace MNN;

namespace utils {
enum class FillType { RANDOM, ZERO };

class RandomEngine {
public:
    static std::mt19937& get() {
        static std::random_device device;
        static std::mt19937 gen(device());
        return gen;
    }
};

template <typename T>
struct RandomGenerator;

template <>
struct RandomGenerator<float> {
    static float generate() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(RandomEngine::get());
    }
};

template <>
struct RandomGenerator<int> {
    static int generate() {
        std::uniform_int_distribution<int> dist(0, 100);
        return dist(RandomEngine::get());
    }
};
} // namespace utils

class LhsPackingTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        return testIndirectionTable1() && testIndirectionTable2() && testWeightConversion();
    }

private:
    bool testIndirectionTable(const ConvParams& params, int batchSize, int inputHeight, int inputWidth) {
        auto outputSize             = params.getOutputSize(inputHeight, inputWidth);
        int outputHeight            = outputSize.height;
        int outputWidth             = outputSize.width;
        std::vector<int> inputShape = {batchSize, inputHeight, inputWidth, params.inputChannel};

        std::vector<float> input(std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<int>()));
        std::vector<float> padValues(params.inputChannel);

        int blockSize = 32;
        auto table    = IndirectionTable<float>(inputShape, params, input.data(), padValues.data(), blockSize);

        bool succ = true;

        // Check the first row
        for (int col = 0; col < blockSize; col++) {
            int oh = col / outputWidth;
            int ow = col % outputWidth;
            int ih = oh * params.strideHeight - params.padTop;
            int iw = ow * params.strideWidth - params.padLeft;

            if (ih < 0 || ih >= inputHeight) {
                succ &= (table.data[col] == padValues.data());
            } else if (iw < 0 || iw >= inputWidth) {
                succ &= (table.data[col] == padValues.data());
            } else {
                int offset = (ih * inputWidth + iw) * params.inputChannel;
                succ &= (table.data[col] == input.data() + offset);
            }
        }
        return succ;
    }

    bool testIndirectionTable1() {
        ConvParams params{
            .inputChannel  = 3,
            .outputChannel = 5,
            .kernelHeight  = 3,
            .kernelWidth   = 2,
            .strideHeight  = 2,
            .strideWidth   = 1,
            .padTop        = 1,
            .padBottom     = 3,
            .padLeft       = 2,
            .padRight      = 1,
            .dilatedHeight = 1,
            .dilatedWidth  = 2,
        };

        int batchSize   = 4;
        int inputHeight = 7;
        int inputWidth  = 5;

        return testIndirectionTable(params, batchSize, inputHeight, inputWidth);
    }

    bool testIndirectionTable2() {
        ConvParams params{
            .inputChannel  = 256,
            .outputChannel = 256,
            .kernelHeight  = 3,
            .kernelWidth   = 3,
            .strideHeight  = 1,
            .strideWidth   = 1,
            .padTop        = 1,
            .padBottom     = 1,
            .padLeft       = 1,
            .padRight      = 1,
            .dilatedHeight = 1,
            .dilatedWidth  = 1,
        };

        int batchSize   = 1;
        int inputHeight = 24;
        int inputWidth  = 24;

        return testIndirectionTable(params, batchSize, inputHeight, inputWidth);
    }

    bool testWeightConversion() {
        std::vector<int> shape = {4, 5, 6, 7};
        int size               = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        std::vector<float> weightSrc(size);
        std::vector<float> weightDst(size);

        for (int i = 0; i < size; i++) {
            weightSrc[i] = i;
        }

        ConvertOIHWToHWIO(weightDst.data(), weightSrc.data(), shape);

        bool succ = true;

        for (int oc = 0; oc < 4; oc++) {
            for (int ic = 0; ic < 5; ic++) {
                for (int h = 0; h < 6; h++) {
                    for (int w = 0; w < 7; w++) {
                        int oo = (h * 7 + w) * 5 * 4 + ic * 4 + oc;
                        int io = oc * 5 * 6 * 7 + ic * 6 * 7 + h * 7 + w;
                        succ &= (weightSrc[io] == weightDst[oo]);
                    }
                }
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(LhsPackingTest, "imatmul/lhs");
#endif
