# 使用预训练模型Finetune
## Finetune原理
深度模型可以看做一种特征提取器，例如卷积神经网络（CNN）可以看做一种视觉特征提取器。但是这种特征提取器需要进行广泛的训练才能避免过拟合数据集，取得较好的泛化性能。如果我们直接搭建一个模型然后在我们自己的小数据集上进行训练的话，很容易过拟合。这时候我们就可以用在一些相似任务的大型数据集上训练得到的模型，在我们自己的小数据集上进行Finetune，即可省去大量训练时间，且获得较好的性能表现。

## 使用场景
例如对于图像分类任务，我们可以使用在ImageNet数据集上训练过的模型如MobilenetV2等，取出其特征提取部分，而替换其最后的分类层（ImageNet有1000类，我们自己的数据集可能只有10类），然后仅对替换之后的分类层进行训练即可。这是因为MobilenetV2的特征提取部分已经得到了较充分的训练，这些特征提取器提取出来的特征对于其他图像是通用的。还有很多其他的任务，例如NLP中可以用在大型语料库上训练过的BERT模型在自己的语料库上进行Finetune。

## MNN Finetune示例
下面以MobilenetV2在自己的4分类小数据集上Finetune为例，演示MNN中Finetune的用法。相关代码在 MNN_ROOT/tools/train/source/demo/mobilenetV2Train.cpp 和 MNN_ROOT/tools/train/source/demo/mobilenetV2Utils.cpp中，可以适当选择大一点的学习率如0.001，加快学习速度
注意，此demo中需要MobilenetV2的MNN模型
```cpp
//  mobilenetV2Train.cpp

class MobilenetV2TransferModule : public Module {
public:
    MobilenetV2TransferModule(const char* fileName) {
        // 读取原始MobilenetV2模型
        auto varMap  = Variable::loadMap(fileName);
        // MobilenetV2的输入节点
        auto input   = Variable::getInputAndOutput(varMap).first.begin()->second;
        // MobilenetV2分类层之前的节点，AveragePooling的输出
        auto lastVar = varMap["MobilenetV2/Logits/AvgPool"];

        // 初始化一个4分类的全连接层，MNN中可以用卷积来表示全连接层
        NN::ConvOption option;
        option.channel = {1280, 4};
        mLastConv      = std::shared_ptr<Module>(NN::Conv(option));

        // 初始化内部特征提取器, 内部提取器设成不需要训练
        mFix.reset(PipelineModule::extract({input}, {lastVar}, false));

        // 注意这里只注册了我们新初始化的4分类全连接层，那么训练时将只更新此4分类全连接层
        registerModel({mLastConv});
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        // 输入一张图片，获得MobilenetV2特征提取器的输出
        auto pool   = mFix->forward(inputs[0]);
        
        // 将上面提取的特征输入到新初始化的4分类层进行分类
        auto result = _Softmax(_Reshape(_Convert(mLastConv->forward(pool), NCHW), {0, -1}));
        
        return {result};
    }
    // MobilenetV2特征提取器，从输入一直到最后一个AveragePooling
    std::shared_ptr<Module> mFix;
    // 重新初始化的4分类全连接层
    std::shared_ptr<Module> mLastConv;
};

class MobilenetV2Transfer : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            std::cout << "usage: ./runTrainDemo.out MobilentV2Transfer /path/to/mobilenetV2Model path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt"
                      << std::endl;
            return 0;
        }

        std::string trainImagesFolder = argv[2];
        std::string trainImagesTxt = argv[3];
        std::string testImagesFolder = argv[4];
        std::string testImagesTxt = argv[5];
        
		// 读取模型，并替换最后一层分类层
        std::shared_ptr<Module> model(new MobilenetV2TransferModule(argv[1]));
		// 进入训练环节
        MobilenetV2Utils::train(model, 4, 0, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);

        return 0;
    }
};
```