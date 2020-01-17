//
//  ScheduleTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/Pipeline.hpp"
#include "core/Schedule.hpp"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;
using namespace std;

struct DataStream {
    ifstream input;
    DataStream(const char* file) {
        // input = std::ifstream(file);	// ifstream doesn't provide a copy constructor (it's deleted),
        input.open(file);
    };

    inline double next() {
        double d = 0.0f;
        input >> d;
        return d;
    }
    ~DataStream() {
        input.close();
    }
};

class FakeSession : public Session {
public:
    FakeSession(const Schedule::ScheduleInfo& info) : Session(info) {
    }

    const std::vector<std::shared_ptr<Pipeline>>& getFakePipelines() const {
        return this->getPipelines();
    }
};

class FakePipeline : public Pipeline {
public:
    FakePipeline(const std::vector<Schedule::PipelineInfo>& info, Backend* backend, Backend* cpuBackend)
        : Pipeline(info, backend, cpuBackend){};

    const std::vector<std::shared_ptr<Unit>>& getFakeUnit() const {
        return this->getUnit();
    }
};

static flatbuffers::Offset<Op> createFlatBufferOp(flatbuffers::FlatBufferBuilder& fbb, string name, int b, int c, int h,
                                                  int w, bool tensorflow) {
    auto dims = fbb.CreateVector(tensorflow ? std::vector<int>({b, h, w, c}) : std::vector<int>({b, c, h, w}));
    auto ib   = InputBuilder(fbb);
    ib.add_dims(dims);
    auto input = ib.Finish();
    auto n     = fbb.CreateString(name);
    auto iv    = fbb.CreateVector(std::vector<int>({0}));
    auto ov    = fbb.CreateVector(std::vector<int>({0}));
    auto op    = OpBuilder(fbb);
    op.add_type(OpType_MIN);
    op.add_name(n);
    op.add_inputIndexes(iv);
    op.add_outputIndexes(ov);
    op.add_main_type(OpParameter_Input);
    op.add_main(flatbuffers::Offset<void>(input.o));
    return op.Finish();
}

static Interpreter* createInterpreter(int b, int c, int h, int w, bool tensorflow) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;
    const std::vector<string> op_names = {"X", "Y", "A", "B", "C", "D", "E", "F"};
    for (string n : op_names) {
        vec.push_back(createFlatBufferOp(fbb, n, b, c, h, w, tensorflow));
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings(op_names);
    if (tensorflow) {
        auto bb = BlobBuilder(fbb);
        bb.add_dataType(DataType_DT_FLOAT);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto flt = bb.Finish();

        std::vector<flatbuffers::Offset<TensorDescribe>> desc;
        TensorDescribeBuilder tdb0(fbb), tdb1(fbb);
        tdb0.add_index(0);
        tdb1.add_index(1);
        tdb0.add_blob(flatbuffers::Offset<Blob>(flt.o));
        tdb1.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb0.Finish());
        desc.push_back(tdb1.Finish());
        auto extras = fbb.CreateVector(desc);
        auto net    = NetBuilder(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        net.add_extraTensorDescribe(extras);
        net.add_sourceType(NetSource_TENSORFLOW);
        fbb.Finish(net.Finish());
    } else {
        auto net = NetBuilder(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        fbb.Finish(net.Finish());
    }
    return Interpreter::createFromBuffer((const char*)fbb.GetBufferPointer(), fbb.GetSize());
}

/*
 * OP X,Y,A,B,C,D,E,F
 * one path expect A->B->C->D
 * multi path expect A->B->C->D->X or X->A->B->C->D
 * */
static void TestScheduleSpec() {
    shared_ptr<Interpreter> instance(createInterpreter(0, 0, 0, 0, false));
    ScheduleConfig conf;
    conf.path.inputs.push_back("A");
    conf.path.outputs.push_back("B");
    conf.path.inputs.push_back("B");
    conf.path.outputs.push_back("C");
    conf.path.inputs.push_back("C");
    conf.path.outputs.push_back("D");

    Session* session         = instance->createSession(conf);
    FakeSession* fakeSession = reinterpret_cast<FakeSession*>(session);

    const std::vector<std::shared_ptr<Pipeline>>& pipelines   = fakeSession->getFakePipelines();
    FakePipeline* fakePipeline                                = reinterpret_cast<FakePipeline*>(pipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& units = fakePipeline->getFakeUnit();
    stringstream ss;
    ss.str("");
    ss.clear();
    for (auto unit : units) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    string rel_str(ss.str());
    string exp_str = "A->B->C->D->";
    int exp_val    = exp_str.compare(rel_str);
    if (0 != exp_val) {
        MNN_ERROR("TestScheduleSpec expect 'A->B->C->D->' output is %s\n", ss.str().c_str());
    }

    std::vector<ScheduleConfig> configs;
    ScheduleConfig conf2;
    conf2.path.inputs.push_back("D");
    conf2.path.outputs.push_back("F");

    /*incorrect path ,will be ignore */
    conf2.path.inputs.push_back("B");

    /*conf, not conf2 ,means X->X*/
    conf.path.outputs.push_back("X");

    conf2.path.inputs.push_back("C");
    conf2.path.outputs.push_back("Y");

    /*real path B->Y C->END ,B->Y incorrect ignored ,*/

    configs.push_back(conf);
    configs.push_back(conf2);

    Session* sessionMulti                                        = instance->createMultiPathSession(configs);
    fakeSession                                                  = reinterpret_cast<FakeSession*>(sessionMulti);
    const std::vector<std::shared_ptr<Pipeline>>& multiPipelines = fakeSession->getFakePipelines();
    fakePipeline = reinterpret_cast<FakePipeline*>(multiPipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& multiUnits = fakePipeline->getFakeUnit();
    FakePipeline* fakePipeline2 = reinterpret_cast<FakePipeline*>(multiPipelines[1].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& multiUnits2 = fakePipeline2->getFakeUnit();
    stringstream ss2;
    ss2.str("");
    ss2.clear();

    ss.str("");
    ss.clear();
    for (auto unit : multiUnits) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }
    for (auto unit : multiUnits2) {
        string name(unit->mOriginOp->name()->c_str());
        ss2 << name << "->";
    }
    rel_str           = ss.str();
    string rel_str2   = ss2.str();
    exp_str           = "A->B->C->D->X->";
    string exp_str1_2 = "X->A->B->C->D->";
    exp_val           = exp_str.compare(rel_str);
    if (0 != exp_val) {
        exp_val = exp_str1_2.compare(rel_str);
    }
    string exp_str2 = "C->D->E->F->";
    int exp_val2    = exp_str2.compare(rel_str2);
    if ((0 != exp_val) || (0 != exp_val2)) {
        MNN_ERROR("TestScheduleSpec expect 'A->B->C->D->X or X->A->B->C->D,C->D->E->F' output is  %s,%s\n",
                  ss.str().c_str(), ss2.str().c_str());
    }
}

/*
 * OP X,Y,A,B,C,D,E,F
 * one path expect A->B->C->D
 * multi path expect A->B->C->D->E->F
 * */
static void TestSchedule() {
    shared_ptr<Interpreter> instance(createInterpreter(0, 0, 0, 0, false));
    ScheduleConfig conf;
    conf.path.inputs.push_back("A");
    conf.path.outputs.push_back("B");
    conf.path.inputs.push_back("B");
    conf.path.outputs.push_back("C");
    conf.path.inputs.push_back("C");
    conf.path.outputs.push_back("D");

    Session* session         = instance->createSession(conf);
    FakeSession* fakeSession = reinterpret_cast<FakeSession*>(session);

    const std::vector<std::shared_ptr<Pipeline>>& pipelines   = fakeSession->getFakePipelines();
    FakePipeline* fakePipeline                                = reinterpret_cast<FakePipeline*>(pipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& units = fakePipeline->getFakeUnit();
    stringstream ss;
    ss.str("");
    ss.clear();
    for (auto unit : units) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    string rel_str(ss.str());
    string exp_str = "A->B->C->D->";
    int exp_val    = exp_str.compare(rel_str);
    if (0 != exp_val) {
        MNN_ERROR("TestSchedule expect 'A->B->C->D->' output is %s\n", rel_str.c_str());
    }

    std::vector<ScheduleConfig> configs;
    ScheduleConfig conf2;
    conf2.path.inputs.push_back("D");
    conf2.path.outputs.push_back("F");

    /*incorrect path ,will be ignore */
    conf2.path.inputs.push_back("B");
    conf2.path.outputs.push_back("X");

    /*incorrect path ,will be ignore */
    conf2.path.inputs.push_back("C");
    conf2.path.outputs.push_back("Y");

    configs.push_back(conf);
    configs.push_back(conf2);

    Session* sessionMulti                                        = instance->createMultiPathSession(configs);
    fakeSession                                                  = reinterpret_cast<FakeSession*>(sessionMulti);
    const std::vector<std::shared_ptr<Pipeline>>& multiPipelines = fakeSession->getFakePipelines();
    fakePipeline = reinterpret_cast<FakePipeline*>(multiPipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& multiUnits = fakePipeline->getFakeUnit();
    FakePipeline* fakePipeline2 = reinterpret_cast<FakePipeline*>(multiPipelines[1].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& multiUnits2 = fakePipeline2->getFakeUnit();
    stringstream ss2;
    ss2.str("");
    ss2.clear();

    ss.str("");
    ss.clear();
    for (auto unit : multiUnits) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    for (auto unit : multiUnits2) {
        string name(unit->mOriginOp->name()->c_str());
        ss2 << name << "->";
    }

    rel_str         = ss.str();
    string rel_str2 = ss2.str();
    exp_str         = "A->B->C->D->";
    exp_val         = exp_str.compare(rel_str);
    string exp_str2 = "D->E->F->";
    int exp_val2    = exp_str2.compare(rel_str2);
    if ((0 != exp_val) || (0 != exp_val2)) {
        MNN_ERROR("TestScheduleMulti expect 'A->B->C->D,D->E->F' output is  %s,%s\n", ss.str().c_str(),
                  ss2.str().c_str());
    }
}

/*
 * OP X,Y,A,B,C,D,E,F inputs A
 * one path expect A->B->C->D->E->F
 * multi path expect A->B->C->D->E->F
 */
static void TestScheduleOneInputHaveBeginNoEnd() {
    shared_ptr<Interpreter> instance(createInterpreter(0, 0, 0, 0, false));
    ScheduleConfig conf;
    conf.path.inputs.push_back("A");

    Session* session         = instance->createSession(conf);
    FakeSession* fakeSession = reinterpret_cast<FakeSession*>(session);

    const std::vector<std::shared_ptr<Pipeline>>& pipelines   = fakeSession->getFakePipelines();
    FakePipeline* fakePipeline                                = reinterpret_cast<FakePipeline*>(pipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& units = fakePipeline->getFakeUnit();
    stringstream ss;
    ss.str("");
    ss.clear();
    for (auto unit : units) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    string rel_str(ss.str());
    string exp_str = "A->B->C->D->E->F->";
    int exp_val    = exp_str.compare(rel_str);
    if (0 != exp_val) {
        MNN_ERROR("TestScheduleOneInputHaveBeginNoEnd expect 'A->B->C->D->E->F->' output is %s\n", rel_str.c_str());
    }

    std::vector<ScheduleConfig> configs;
    ScheduleConfig conf2;
    conf2.path.inputs.push_back("A");

    configs.push_back(conf);
    configs.push_back(conf2);

    Session* sessionMulti                                        = instance->createMultiPathSession(configs);
    fakeSession                                                  = reinterpret_cast<FakeSession*>(sessionMulti);
    const std::vector<std::shared_ptr<Pipeline>>& multiPipelines = fakeSession->getFakePipelines();
    fakePipeline = reinterpret_cast<FakePipeline*>(multiPipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& multiUnits = fakePipeline->getFakeUnit();
    ss.str("");
    ss.clear();
    for (auto unit : multiUnits) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    rel_str = ss.str();
    exp_str = "A->B->C->D->E->F->";
    exp_val = exp_str.compare(rel_str);
    if (0 != exp_val) {
        MNN_ERROR("TestScheduleOneInputHaveBeginNoEnd expect 'A->B->C->D->E->F' output is  %s\n", rel_str.c_str());
    }
}

/*
 * OP X,Y,A,B,C,D,E,F inputs A D
 * one path expect A->B->C->D->E->F
 * multi path expect A->B->C->D->E->F
 */
static void TestScheduleMultiInputsHaveBeginNoEnd() {
    shared_ptr<Interpreter> instance(createInterpreter(0, 0, 0, 0, false));
    ScheduleConfig conf;
    conf.path.inputs.push_back("A");
    conf.path.inputs.push_back("D");

    Session* session         = instance->createSession(conf);
    FakeSession* fakeSession = reinterpret_cast<FakeSession*>(session);

    const std::vector<std::shared_ptr<Pipeline>>& pipelines   = fakeSession->getFakePipelines();
    FakePipeline* fakePipeline                                = reinterpret_cast<FakePipeline*>(pipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& units = fakePipeline->getFakeUnit();
    stringstream ss;
    ss.str("");
    ss.clear();
    for (auto unit : units) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    string rel_str(ss.str());
    string exp_str = "A->B->C->D->E->F->";
    int exp_val    = exp_str.compare(rel_str);
    if (0 != exp_val) {
        MNN_ERROR("TestScheduleMultiInputHaveBeginNoEnd expect 'A->B->C->D->E->F->' output is %s\n", rel_str.c_str());
    }

    std::vector<ScheduleConfig> configs;
    ScheduleConfig conf2;
    conf2.path.inputs.push_back("A");
    conf2.path.inputs.push_back("D");

    configs.push_back(conf);
    configs.push_back(conf2);

    Session* sessionMulti                                        = instance->createMultiPathSession(configs);
    fakeSession                                                  = reinterpret_cast<FakeSession*>(sessionMulti);
    const std::vector<std::shared_ptr<Pipeline>>& multiPipelines = fakeSession->getFakePipelines();
    fakePipeline = reinterpret_cast<FakePipeline*>(multiPipelines[0].get());
    const std::vector<std::shared_ptr<Pipeline::Unit>>& multiUnits = fakePipeline->getFakeUnit();
    ss.str("");
    ss.clear();
    for (auto unit : multiUnits) {
        string name(unit->mOriginOp->name()->c_str());
        ss << name << "->";
    }

    rel_str = ss.str();
    exp_str = "A->B->C->D->E->F->";
    exp_val = exp_str.compare(rel_str);
    if (0 != exp_val) {
        MNN_ERROR("TestScheduleMultiInputHaveBeginNoEnd expect 'A->B->C->D->E->F' output is  %s\n", rel_str.c_str());
    }
}

// set squeezenetv1.1 model file path,from project MNNModel
static const string const_model_file  = "resource/model/SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn";
static const string const_input_file  = "resource/model/SqueezeNet/input.txt";
static const string const_output_file = "resource/model/SqueezeNet/v1.1/expect.txt";
static string model_file              = "../" + const_model_file;
static string input_file              = "../" + const_input_file;
static string output_file             = "../" + const_output_file;
static double threshold               = 1.0e-4;

/* squeezenetv1.1 op orders

op name = conv1
op name = pool1
op name = fire2/squeeze1x1
op name = fire2/expand1x1
op name = fire2/expand3x3
op name = fire2/concat
op name = fire3/squeeze1x1
op name = fire3/expand1x1
op name = fire3/expand3x3
op name = fire3/concat
op name = pool3
op name = fire4/squeeze1x1
op name = fire4/expand1x1
op name = fire4/expand3x3
op name = fire4/concat
op name = fire5/squeeze1x1
op name = fire5/expand1x1
op name = fire5/expand3x3
op name = fire5/concat
op name = pool5
op name = fire6/squeeze1x1
op name = fire6/expand1x1
op name = fire6/expand3x3
op name = fire6/concat
op name = fire7/squeeze1x1
op name = fire7/expand1x1
op name = fire7/expand3x3
op name = fire7/concat
op name = fire8/squeeze1x1
op name = fire8/expand1x1
op name = fire8/expand3x3
op name = fire8/concat
op name = fire9/squeeze1x1
op name = fire9/expand1x1
op name = fire9/expand3x3
op name = fire9/concat
op name = conv10
op name = pool10
op name = prob

*/

struct free_delete {
    void operator()(void* x) {
        free(x);
    }
};

static MNN::Tensor* createTensor(const MNN::Tensor* shape, const char* path) {
    std::ifstream stream(path);
    if (stream.fail()) {
        return NULL;
    }

    auto result = new MNN::Tensor(shape, shape->getDimensionType());
    auto data   = result->host<float>();
    for (int i = 0; i < result->elementSize(); ++i) {
        double temp = 0.0f;
        stream >> temp;
        data[i] = temp;
    }
    stream.close();
    return result;
}

static void TestSqueezeNet() {
    const shared_ptr<Interpreter> net(Interpreter::createFromFile(model_file.c_str()));
    ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    Session* session = net->createSession(config);

    Tensor* inputTensor = net->getSessionInput(session, NULL);
    const shared_ptr<Tensor> givenTensor(createTensor(inputTensor, input_file.c_str()));
    if (!givenTensor) {
        MNN_ERROR("[FAIL] TestSqueezeNetFailed to open input file %s.\n", input_file.c_str());
        return;
    }
    net->getBackend(session, inputTensor)->onCopyBuffer(givenTensor.get(), inputTensor);

    Tensor* outputTensor = net->getSessionOutput(session, NULL);
    shared_ptr<Tensor> expectTensor(createTensor(outputTensor, output_file.c_str()));
    if (!expectTensor.get()) {
        MNN_ERROR("[FAIL] TestSqueezeNetFailed to open output file %s.\n", input_file.c_str());
        return;
    }

    net->runSession(session);

    // compare output with expect
    bool correct = TensorUtils::compareTensors(outputTensor, expectTensor.get(), threshold, true);
    if (!correct) {
        MNN_ERROR("[FAIL] TestSqueezeNet %s fail to run\n", model_file.c_str());
    }
}

static void TestSqueezeNetOnePath() {
    const shared_ptr<Interpreter> net(Interpreter::createFromFile(model_file.c_str()));
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.path.inputs.push_back("conv1");
    Session* session = net->createSession(config);

    Tensor* inputTensor = net->getSessionInput(session, NULL);
    const shared_ptr<Tensor> givenTensor(createTensor(inputTensor, input_file.c_str()));
    if (!givenTensor) {
        MNN_ERROR("[FAIL] TestSqueezeNetOnePath to open input file %s.\n", input_file.c_str());
        return;
    }
    net->getBackend(session, inputTensor)->onCopyBuffer(givenTensor.get(), inputTensor);

    Tensor* outputTensor = net->getSessionOutput(session, NULL);
    shared_ptr<Tensor> expectTensor(createTensor(outputTensor, output_file.c_str()));
    if (!expectTensor.get()) {
        MNN_ERROR("[FAIL] TestSqueezeNetOnePath to open output file %s.\n", input_file.c_str());
        return;
    }

    net->runSession(session);

    // compare output with expect
    bool correct = TensorUtils::compareTensors(outputTensor, expectTensor.get(), threshold, true);
    if (!correct) {
        MNN_ERROR("[FAIL] TestSqueezeNetOnePath %s fail to run\n", model_file.c_str());
    }
}

static void TestSqueezeNetOnePathFailed() {
    const shared_ptr<Interpreter> net(Interpreter::createFromFile(model_file.c_str()));
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.path.inputs.push_back("conv1");
    config.path.outputs.push_back("pool1");
    Session* session = net->createSession(config);

    Tensor* inputTensor = net->getSessionInput(session, NULL);
    const shared_ptr<Tensor> givenTensor(createTensor(inputTensor, input_file.c_str()));
    if (!givenTensor) {
        MNN_ERROR("[FAIL] TestSqueezeNetOnePathFailed to open input file %s.\n", input_file.c_str());
        return;
    }
    net->getBackend(session, inputTensor)->onCopyBuffer(givenTensor.get(), inputTensor);

    Tensor* outputTensor = net->getSessionOutput(session, NULL);
    shared_ptr<Tensor> expectTensor(createTensor(outputTensor, output_file.c_str()));
    if (!expectTensor.get()) {
        MNN_ERROR("[FAIL] TestSqueezeNetOnePathFailed to open output file %s.\n", input_file.c_str());
        return;
    }

    net->runSession(session);

    // compare output with expect
    bool correct = TensorUtils::compareTensors(outputTensor, expectTensor.get(), threshold, true);
    if (correct) {
        MNN_ERROR("[FAIL] TestSqueezeNetOnePath %s fail to run\n", model_file.c_str());
    }
}

static void TestScheduleSqueezeNetMultiPathFailed() {
    const shared_ptr<Interpreter> net(Interpreter::createFromFile(model_file.c_str()));

    ScheduleConfig conf1;
    conf1.type = MNN_FORWARD_CPU;
    conf1.path.inputs.push_back("conv1");
    conf1.path.outputs.push_back("pool5");
    ScheduleConfig conf2;
    conf2.path.inputs.push_back("fire6/squeeze1x1");
    conf2.path.outputs.push_back("conv10");
    conf2.type = MNN_FORWARD_CPU;

    vector<ScheduleConfig> configs;
    configs.push_back(conf1);
    configs.push_back(conf2);
    Session* session = net->createMultiPathSession(configs);

    Tensor* inputTensor = net->getSessionInput(session, NULL);
    const shared_ptr<Tensor> givenTensor(createTensor(inputTensor, input_file.c_str()));
    if (!givenTensor) {
        MNN_ERROR("[FAIL] TestScheduleSqueezeNetMultiPathFailed to open input file %s.\n", input_file.c_str());
        return;
    }
    net->getBackend(session, inputTensor)->onCopyBuffer(givenTensor.get(), inputTensor);

    Tensor* outputTensor = net->getSessionOutput(session, NULL);
    const shared_ptr<Tensor> expectTensor(createTensor(outputTensor, output_file.c_str()));
    if (!expectTensor.get()) {
        MNN_ERROR("[FAIL] TestScheduleSqueezeNetMultiPathFailed to open output file %s.\n", input_file.c_str());
        return;
    }

    net->runSession(session);

    // compare output with expect
    bool correct = MNN::TensorUtils::compareTensors(outputTensor, expectTensor.get(), threshold, true);
    if (correct) {
        MNN_ERROR("[FAIL] TestScheduleSqueezeNetMultiPathFailed %s fail to run\n", model_file.c_str());
    }
}

static void TestScheduleSqueezeNetMultiPath() {
    const shared_ptr<Interpreter> net(Interpreter::createFromFile(model_file.c_str()));

    ScheduleConfig conf1;
    conf1.type = MNN_FORWARD_CPU;
    conf1.path.inputs.push_back("conv1");
    conf1.path.outputs.push_back("pool5");
    ScheduleConfig conf2;
    conf2.path.inputs.push_back("fire6/squeeze1x1");
    conf2.type = MNN_FORWARD_CPU;

    vector<ScheduleConfig> configs;
    configs.push_back(conf1);
    configs.push_back(conf2);
    Session* session = net->createMultiPathSession(configs);

    Tensor* inputTensor = net->getSessionInput(session, NULL);
    const shared_ptr<Tensor> givenTensor(createTensor(inputTensor, input_file.c_str()));
    if (!givenTensor) {
        MNN_ERROR("[FAIL] TestSqueezeNetFailed to open input file %s.\n", input_file.c_str());
        return;
    }
    net->getBackend(session, inputTensor)->onCopyBuffer(givenTensor.get(), inputTensor);

    Tensor* outputTensor = net->getSessionOutput(session, NULL);
    const shared_ptr<Tensor> expectTensor(createTensor(outputTensor, output_file.c_str()));
    if (!expectTensor.get()) {
        MNN_ERROR("[FAIL] TestSqueezeNetFailed to open output file %s.\n", input_file.c_str());
        return;
    }

    net->runSession(session);

    // compare output with expect
    bool correct = MNN::TensorUtils::compareTensors(outputTensor, expectTensor.get(), threshold, true);
    if (!correct) {
        MNN_ERROR("[FAIL] TestSqueezeNet %s fail to run\n", model_file.c_str());
    }
}

class ScheduleTest : public MNNTestCase {
public:
    virtual bool run();
    ScheduleTest() {
    }
    virtual ~ScheduleTest() {
    }
};

bool ScheduleTest::run() {
    TestSchedule();
    TestScheduleSpec();
    TestScheduleOneInputHaveBeginNoEnd();
    TestScheduleMultiInputsHaveBeginNoEnd();
    bool squeezeNetCont = true;
    string path_join    = "../";
    string path         = path_join + const_model_file;
#if defined(_MSC_VER)
    if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(path.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
        path_join = "./";
        path      = path_join + const_model_file;
        if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(path.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
#else
    if (-1 == access(path.c_str(), F_OK)) {
        path_join = "./";
        path      = path_join + const_model_file;
        if (-1 == access(path.c_str(), F_OK)) {
#endif
            squeezeNetCont = false;
            MNN_ERROR("[FAIL] TestSqueezeNet %s fail to run.Model file not found\n", const_model_file.c_str());
        }
    }
    if (squeezeNetCont) {
        model_file  = path_join + const_model_file;
        input_file  = path_join + const_input_file;
        output_file = path_join + const_output_file;

        TestSqueezeNet();
        TestSqueezeNetOnePath();
        TestSqueezeNetOnePathFailed();
        TestScheduleSqueezeNetMultiPath();
        TestScheduleSqueezeNetMultiPathFailed();
    }
    return true;
}

MNNTestSuiteRegister(ScheduleTest, "engine/schedule");
