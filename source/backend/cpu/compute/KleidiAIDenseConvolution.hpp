#if MNN_KLEIDIAI_ENABLED

#ifndef KleidiAIDenseConvolution_hpp
#define KleidiAIDenseConvolution_hpp

#include "ConvolutionTiledExecutor.hpp"
#include "backend/cpu/CPUConvolution.hpp"

namespace MNN {
struct ConvParams {
    int inputChannel;
    int outputChannel;
    int kernelHeight;
    int kernelWidth;
    int strideHeight;
    int strideWidth;
    int padTop;
    int padBottom;
    int padLeft;
    int padRight;
    int dilatedHeight;
    int dilatedWidth;

    struct Size2D {
        int height;
        int width;
    };

    Size2D getOutputSize(int inputHeight, int inputWidth) const {
        auto kernelSizeWithDilated = [](int kernel, int dilated) { return kernel + (kernel - 1) * (dilated - 1); };
        auto outputSize            = [](int input, int pad1, int pad2, int kernel, int stride) {
            int t = (input + pad1 + pad2 - kernel);
            return t / stride + 1;
        };

        int dilatedKernelHeight = kernelSizeWithDilated(kernelHeight, dilatedHeight);
        int dilatedKernelWidth  = kernelSizeWithDilated(kernelWidth, dilatedWidth);

        int outputHeight = outputSize(inputHeight, padTop, padBottom, dilatedKernelHeight, strideHeight);
        int outputWidth  = outputSize(inputHeight, padLeft, padRight, dilatedKernelWidth, strideWidth);

        return {outputHeight, outputWidth};
    }
};

template <typename T>
struct IndirectionTable {
    std::vector<const void*> data;
    int height;
    int width;
    int blockSize;

    /// Creates an indirection table for LHS packing.
    ///
    /// When implementing convolution via matrix multiplication, we need to
    /// transform the input and weight tensors into matrices. This transformation
    /// for the input is typically referred to as `im2col`. The resulting matrix has
    /// dimensions:
    /// - Rows: batch * output_height * output_width
    /// - Columns: input_channels * kernel_height * kernel_width
    ///
    /// The indirection table stores the starting addresses of all these chunks in
    /// the input tensor. For cases where padding is applied, it stores pointers
    /// directly to the padded buffer. Note that the length of the padding buffer
    /// must match the number of input channels.
    ///
    /// The indirection table stores the starting addresses of all these chunks in
    /// the input tensor. Furthermore, LHS packing also requires a transpose over
    /// every `M_STEP` rows to optimize data layout for computation.
    ///
    /// @param[in] shape The NHWC input shape
    /// @param[in] params The parameters of convolution
    /// @param[in] input The raw pointer for the input tensor
    /// @param[in] padValues The raw pointer for the pad tensor
    /// @param[in] blockSize The block size for the transpose
    ///
    /// @return The indirection table ready for lhs packing.
    IndirectionTable(const std::vector<int>& shape, const ConvParams& params, const T* input, const T* padValues,
                     const int blockSize);

    ~IndirectionTable() = default;

    /// To compute the offset after blocking of blockSize.
    ///
    /// @param[in] row The row index
    /// @param[in] col The col index
    /// @param[in] width The table column count
    /// @param[in] block The block size
    ///
    /// @return The offset in blocking table
    int getReorderedOffset(int row, int col, int width, int block) {
        int c = row % block;
        int r = row / block * width + col;
        return r * block + c;
    }
};

template <typename T>
IndirectionTable<T>::IndirectionTable(const std::vector<int>& shape, const ConvParams& params, const T* input,
                                      const T* padValues, const int blockSize) {
    int batchSize    = shape[0];
    int inputChannel = shape[3];
    int inputHeight  = shape[1];
    int inputWidth   = shape[2];

    int elementCount = batchSize * inputChannel * inputHeight * inputWidth;
    auto outputSize  = params.getOutputSize(inputHeight, inputWidth);
    int outputHeight = outputSize.height;
    int outputWidth  = outputSize.width;

    int rowCount = batchSize * outputHeight * outputWidth;
    int colCount = params.kernelHeight * params.kernelWidth;

    this->data.resize((rowCount + blockSize - 1) / blockSize * blockSize * colCount);
    this->height    = rowCount;
    this->width     = colCount;
    this->blockSize = blockSize;

    for (int i = 0; i < this->data.size(); i++) {
        this->data[i] = nullptr;
    }

    for (int b = 0; b < batchSize; b++) {
        for (int h = 0; h < outputSize.height; h++) {
            for (int w = 0; w < outputSize.width; w++) {
                int inputRow = h * params.strideHeight - params.padTop;
                int inputCol = w * params.strideWidth - params.padLeft;

                for (int kh = 0; kh < params.kernelHeight; kh++) {
                    // Every row of im2col resulting matrix $kernel height * kernel width$
                    // chunks. So indirection table has relevant values, which point to the
                    // relevant chunk. The `tableRow` and `tableCol` is the row and column
                    // of the table not transposed.
                    int tableRow = b * outputHeight * outputWidth + h * outputWidth + w;
                    int tableCol = kh * params.kernelWidth;

                    int inputRowPrime    = inputRow + kh * params.dilatedHeight;
                    int inputOffsetStart = b * inputHeight * inputWidth + inputRowPrime * inputWidth;
                    if (inputRowPrime >= 0 && inputRowPrime < inputHeight) {
                        for (int kw = 0; kw < params.kernelWidth; kw++) {
                            int tableOffset   = getReorderedOffset(tableRow, tableCol + kw, colCount, blockSize);
                            int inputColPrime = inputCol + kw * params.dilatedWidth;
                            if (inputColPrime >= 0 && inputColPrime < inputWidth) {
                                int inputOffset = (inputOffsetStart + inputColPrime) * inputChannel;
                                assert(inputOffset < elementCount);
                                assert(tableOffset < this->data.size());
                                this->data[tableOffset] = input + inputOffset;
                            } else {
                                assert(tableOffset < this->data.size());
                                this->data[tableOffset] = padValues;
                            }
                        }
                    } else {
                        for (int kw = 0; kw < params.kernelWidth; kw++) {
                            int tableOffset = getReorderedOffset(tableRow, tableCol + kw, colCount, blockSize);
                            assert(tableOffset < this->data.size());
                            this->data[tableOffset] = padValues;
                        }
                    }
                }
            }
        }
    }
}

template <typename DstT, typename SrcT>
static void ConvertOIHWToHWIO(DstT* dst, const SrcT* src, const std::vector<int>& shape) {
    assert(shape.size() == 4);
    int height        = shape[2];
    int width         = shape[3];
    int outputChannel = shape[0];
    int inputChannel  = shape[1];

    int spatialSize = height * width;
    for (int oc = 0; oc < outputChannel; oc++) {
        for (int ic = 0; ic < inputChannel; ic++) {
            for (int s = 0; s < spatialSize; s++) {
                int inputOffset  = oc * inputChannel * spatialSize + ic * spatialSize + s;
                int outputOffset = s * inputChannel * outputChannel + ic * outputChannel + oc;

                // TODO Check the force conversion.
                dst[outputOffset] = (DstT)(src[inputOffset]);
            }
        }
    }
}

class KleidiAIDenseConvolutionImpl : public ConvolutionTiledImpl {
public:
    KleidiAIDenseConvolutionImpl(const Convolution2DCommon *common, Backend *b,
                                 CPUConvolution::Resource *resource = nullptr)
        : ConvolutionTiledImpl(common, b) {
        mResource = resource;
    }
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ~KleidiAIDenseConvolutionImpl() = default;
    virtual void getPackParameter(int *eP, int *lP, int *hP, const CoreFunctions *core) override {}

private:
    Tensor mOutputNHWC;
    Tensor mInputNHWC;
    Tensor mPadBuffer;
};

class KleidiAIDenseConvolution : public ConvolutionTiledExecutor {
public:
    KleidiAIDenseConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize,
                             std::shared_ptr<ConvolutionCommon::Int8Common>);

    KleidiAIDenseConvolution(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon *common,
                             Backend *b);
    virtual ~KleidiAIDenseConvolution();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend *bn, const Op *op, Execution **dst) override;
    void initWeight(float *dest, const float *source, float *cache, int depth, int outputCount, int kernelSize,
                    const CoreFunctions *function);

protected:
    std::shared_ptr<KleidiAIDenseConvolutionImpl> mProxy;
};

class KleidiAIDenseConvolutionMultiInput : public Execution {
public:
    KleidiAIDenseConvolutionMultiInput(const Convolution2DCommon *common, Backend *b) : Execution(b) {
        mProxy.reset(new KleidiAIDenseConvolutionImpl(common, b));
    }
    virtual ~KleidiAIDenseConvolutionMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempWeightCache;
    std::shared_ptr<Tensor> mTempBias;
    std::shared_ptr<KleidiAIDenseConvolutionImpl> mProxy;
    std::vector<Tensor *> mInputs;
};
} // namespace MNN

#endif /* KleidiAIDenseConvolution_hpp */
#endif
