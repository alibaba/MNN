//
//  BlstmComputerTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/05/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef OPEN_BI_LSTM_OLD
#include <memory>

#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/BlstmComputer.hpp"
#include "core/Backend.hpp"

using std::shared_ptr;

using namespace MNN;

namespace BlstmComputerTest {
shared_ptr<Tensor> createTensorFromData(const vector<int> &shape, float *data, Tensor::DimensionType dimType,
                                        CPUBackend *bn) {
    auto srcTensor = shared_ptr<Tensor>(Tensor::create(shape, halide_type_of<float>(), data, Tensor::CAFFE));
    auto tgtTensor = shared_ptr<Tensor>(Tensor::createDevice<float>(shape, dimType));
    bn->onAcquireBuffer(tgtTensor.get(), Backend::DYNAMIC);
    bn->onCopyBuffer(srcTensor.get(), tgtTensor.get());
    return tgtTensor;
}

shared_ptr<Tensor> createAndRun(int timesteps, int batch, int inDim, int stateDim, bool bidirectional, float *input,
                                float *weights, float *initH, float *initC, vector<int> lengths,
                                Tensor::DimensionType dimType = Tensor::CAFFE) {
    auto creator = MNNGetExtraRuntimeCreator((MNNForwardType)0);
    std::shared_ptr<Runtime> runtime;
    Backend::Info info;
    info.type = (MNNForwardType)0;
    runtime.reset(creator->onCreate(info));
    auto backend = shared_ptr<CPUBackend>((CPUBackend *)(runtime->onCreate()));

    auto inTensor = createTensorFromData(vector<int>{batch, timesteps, inDim}, input, dimType, backend.get());

    vector<shared_ptr<Tensor>> weightTensors = {};
    vector<shared_ptr<Tensor>> initHs        = {};
    vector<shared_ptr<Tensor>> initCs        = {};
    int offside                              = 0;
    int s_offside                            = 0;
    for (int i = 0; i < (bidirectional ? 2 : 1); i++) {
        // Wi, Wn, Wf, Wo
        for (int j = 0; j < 4; j++) {
            weightTensors.push_back(
                createTensorFromData(vector<int>{inDim, stateDim}, weights + offside, dimType, backend.get()));
            offside += stateDim * inDim;
        }
        // Ui, Un, Uf, Uo
        for (int j = 0; j < 4; j++) {
            weightTensors.push_back(
                createTensorFromData(vector<int>{stateDim, stateDim}, weights + offside, dimType, backend.get()));
            offside += stateDim * stateDim;
        }
        // Bi, Bn, Bf, Bo
        for (int j = 0; j < 4; j++) {
            weightTensors.push_back(
                createTensorFromData(vector<int>{stateDim}, weights + offside, dimType, backend.get()));

            offside += stateDim;
        }

        if (initH) {
            initHs.push_back(
                createTensorFromData(vector<int>{batch, stateDim}, initH + s_offside, dimType, backend.get()));
        }
        if (initC) {
            initCs.push_back(
                createTensorFromData(vector<int>{batch, stateDim}, initC + s_offside, dimType, backend.get()));
        }
        s_offside += batch * stateDim;
    }

    auto blstm = BlstmComputer(inDim, stateDim, bidirectional, backend.get());
    blstm.importWeights(weightTensors);
    blstm.onResize(timesteps, batch);
    blstm.onExecute(inTensor.get(), lengths, initHs, initCs);

    backend->onReleaseBuffer(inTensor.get(), Backend::DYNAMIC);
    for (int i = 0; i < weightTensors.size(); i++) {
        backend->onReleaseBuffer(weightTensors[i].get(), Backend::DYNAMIC);
    }
    for (int i = 0; i < initHs.size(); i++) {
        backend->onReleaseBuffer(initHs[i].get(), Backend::DYNAMIC);
    }
    for (int i = 0; i < initCs.size(); i++) {
        backend->onReleaseBuffer(initCs[i].get(), Backend::DYNAMIC);
    }

    return blstm.output();
}

} // namespace BlstmComputerTest

class BlstmComputerTestNormal : public MNNTestCase {
public:
    virtual bool run(int precision) {
        int timesteps     = 2;
        int batch         = 2;
        int inDim         = 4;
        int stateDim      = 2;
        int bidirectional = true;

        float input[] = {0.44579449, 0.27169554, 0.65260875, 0.31565022, 0.90375165, 0.06445987,
                         0.5355081,  0.00180581, 0.14474092, 0.73850643, 0.33908029, 0.02118387,
                         0.42463244, 0.91618846, 0.16352628, 0.16589524};

        // weights + bias
        // (inDim * stateDim + stateDim * stateDim + stateDim) * 4 * (bidirectional
        // ? 2 : 1)
        float weights[] = {
            0.2238575,  0.48818177, 0.68244838, 0.13634153, 0.66256713, 0.62072643, 0.77099263, 0.525549,   0.98192705,
            0.07248594, 0.86286398, 0.15345954, 0.06705897, 0.87608419, 0.75358085, 0.05641118, 0.95627635, 0.96339125,
            0.47342446, 0.92175236, 0.40918123, 0.43998353, 0.11528957, 0.58670112, 0.14460955, 0.10411315, 0.81912501,
            0.07275662, 0.55807365, 0.89699957, 0.51889823, 0.25331806, 0.66381276, 0.22328322, 0.6877316,  0.92511827,
            0.38725914, 0.04857563, 0.89930032, 0.11227703, 0.0540831,  0.56100574, 0.30010248, 0.94735647, 0.95685347,
            0.09243085, 0.20552173, 0.03578646, 0.76262415, 0.91767524, 0.97650681, 0.35745307, 0.02309165, 0.26921557,
            0.54497556, 0.67020231, 0.55593737, 0.80142984, 0.81460252, 0.31087914, 0.26101857, 0.79632238, 0.94220189,
            0.08358934, 0.1308584,  0.01036083, 0.36453937, 0.30837561, 0.57286502, 0.45081482, 0.21734892, 0.25597718,
            0.51861896, 0.74332179, 0.54929558, 0.17185688, 0.61304193, 0.78736534, 0.98097528, 0.12290096, 0.63355095,
            0.29947352, 0.98152295, 0.63781417, 0.90260405, 0.08585729, 0.21135987, 0.41743537, 0.20267783, 0.80777418,
            0.03285213, 0.9306145,  0.78912835, 0.77490629, 0.7590748,  0.78981628, 0.45081439, 0.95330662, 0.62282867,
            0.38536004, 0.88046332, 0.5811635,  0.84273814, 0.5468788,  0.15999526, 0.95592125, 0.02159869, 0.06027197,
            0.694669,   0.9606723,  0.07577876, 0.08770169};

        // initH
        float initH[] = {0.62370589, 0.41819492, 0.24196935, 0.2528544, 0.74764926, 0.28984179, 0.9113539, 0.06682115};

        float *initC = nullptr;

        // lengths
        vector<int> lengths = {1, 2};

        // output
        float expected[] = {0.63731168, 0.50447114, 0.54004276, 0.51610641, 0.0,        0.0,
                            0.0,        0.0,        0.57669216, 0.3799657,  0.76510281, 0.70311715,
                            0.81803278, 0.5823984,  0.55451016, 0.5276197};

        auto output = BlstmComputerTest::createAndRun(timesteps, batch, inDim, stateDim, bidirectional, input, weights,
                                                      initH, initC, lengths);

        if (!checkVector<float>(output->host<float>(), expected, (bidirectional ? 2 : 1) * stateDim * batch, 0.00001)) {
            MNN_ERROR("BlstmComputerTestNormal failed!\n");
        }

        return true;
    }
};

class BlstmComputerTestUnidirection : public MNNTestCase {
public:
    virtual bool run(int precision) {
        int timesteps     = 3;
        int batch         = 2;
        int inDim         = 4;
        int stateDim      = 3;
        int bidirectional = false;

        float input[] = {0.07063506, 0.65430983, 0.56112311, 0.5369814,  0.80502379, 0.79027412,
                         0.63801155, 0.92361085, 0.99655268, 0.16573212, 0.69464267, 0.35013,
                         0.98724841, 0.36551049, 0.59502615, 0.67222897, 0.3225175,  0.8318001,
                         0.7885656,  0.41313802, 0.26528233, 0.84895828, 0.13094788, 0.53424511};

        // weights + bias
        // (inDim * stateDim + stateDim * stateDim + stateDim) * 4 * (bidirectional
        // ? 2 : 1)
        float weights[] = {
            0.50859157, 0.42032549, 0.64586107, 0.11835709, 0.73791861, 0.91185463, 0.7794255,  0.41338563, 0.63069258,
            0.85007543, 0.92562638, 0.97180915, 0.10553475, 0.28693475, 0.28149734, 0.41888574, 0.817851,   0.58375121,
            0.63028485, 0.1825274,  0.58661691, 0.21433593, 0.52044292, 0.55795387, 0.39298492, 0.69328593, 0.06190795,
            0.28433201, 0.03464289, 0.15658443, 0.34895989, 0.62446767, 0.653802,   0.86148335, 0.73471965, 0.37505477,
            0.09596098, 0.31304082, 0.69340192, 0.35471248, 0.98211136, 0.6882594,  0.35345624, 0.12996293, 0.24033034,
            0.5551432,  0.04605984, 0.65338722, 0.96682094, 0.9017795,  0.61415741, 0.13297137, 0.98512678, 0.98878332,
            0.64480182, 0.9701902,  0.67724349, 0.18685954, 0.39824749, 0.16005411, 0.24644767, 0.9290525,  0.48801532,
            0.78801746, 0.36831906, 0.52040886, 0.0618416,  0.44006301, 0.19891771, 0.58355389, 0.46013828, 0.53544692,
            0.95864813, 0.856201,   0.68847609, 0.14422627, 0.91650289, 0.62196533, 0.40004766, 0.19860889, 0.11448989,
            0.48439903, 0.7697705,  0.68573039, 0.66125425, 0.31255988, 0.86843112, 0.960107,   0.50405756, 0.98466391,
            0.07950028, 0.49837566, 0.86844674, 0.99373642, 0.54659072, 0.03061445};

        float *initH = nullptr;

        float initC[] = {0.14791486, 0.15031535, 0.83345849, 0.05546057, 0.03556074, 0.31749688};

        // lengths
        vector<int> lengths = {3, 2};

        // output
        float expected[] = {0.6061953,  0.54675685, 0.67327551, 0.88831592, 0.8969134,  0.92849427,
                            0.91633294, 0.93643603, 0.92327956, 0.61217156, 0.53396435, 0.68469396,
                            0.86613871, 0.88406268, 0.87251095, 0.0,        0.0,        0.0};

        auto output = BlstmComputerTest::createAndRun(timesteps, batch, inDim, stateDim, bidirectional, input, weights,
                                                      initH, initC, lengths, Tensor::CAFFE_C4);

        if (!checkVector<float>(output->host<float>(), expected, (bidirectional ? 2 : 1) * stateDim * batch, 0.00001)) {
            MNN_ERROR("BlstmComputerTestUnidirection failed!\n");
        }
        return true;
    }
};

class BlstmComputerTestNC4HW4 : public MNNTestCase {
public:
    virtual bool run(int precision) {
        int timesteps     = 2;
        int batch         = 1;
        int inDim         = 3;
        int stateDim      = 2;
        int bidirectional = true;

        float input[] = {
            0.75869048, 0.88609129, 0.8798418, 0.25742624, 0.99163137, 0.88838347,
        };

        // weights + bias
        // (inDim * stateDim + stateDim * stateDim + stateDim) * 4 * (bidirectional
        // ? 2 : 1)
        float weights[] = {
            0.45613769, 0.27979277, 0.83805803, 0.03504496, 0.96313684, 0.40684258, 0.20054775, 0.75492125, 0.83246728,
            0.81082328, 0.48878888, 0.97353006, 0.73899017, 0.52946552, 0.80860641, 0.82713619, 0.92284315, 0.75735167,
            0.83212562, 0.66530174, 0.36496179, 0.82550771, 0.99142771, 0.23734743, 0.59684077, 0.22638516, 0.17868921,
            0.10721502, 0.07456506, 0.43853882, 0.27990261, 0.64133078, 0.61913749, 0.29701733, 0.02623903, 0.10291185,
            0.06500492, 0.45540884, 0.55719567, 0.85839626, 0.35564447, 0.92474717, 0.3348158,  0.25475312, 0.43572161,
            0.93678135, 0.73704404, 0.2207677,  0.61488313, 0.20509485, 0.67676218, 0.67299135, 0.24450207, 0.71503865,
            0.20413044, 0.6166439,  0.27392602, 0.21333179, 0.61642687, 0.89389671, 0.60646685, 0.59487045, 0.95809009,
            0.63808471, 0.20573288, 0.57610769, 0.79254407, 0.7403801,  0.66553459, 0.31707155, 0.91307493, 0.83930246,
            0.78909002, 0.7547809,  0.85268649, 0.90179711, 0.8661858,  0.44489435, 0.95849377, 0.399035,   0.83782053,
            0.37598029, 0.51351171, 0.74813804, 0.6088315,  0.28641813, 0.22747814, 0.69578654, 0.55833504, 0.45369673,
            0.28346377, 0.1549269,  0.44708032, 0.23423822, 0.84319764, 0.38456204};

        // initH
        float initH[] = {0.42454145, 0.60951945, 0.46172789, 0.20042209};

        float initC[] = {
            0.88764419,
            0.77268416,
            0.55927411,
            0.16463758,
        };

        // lengths
        vector<int> lengths = {2};

        // output
        float expected[] = {0.89114337, 0.83993938, 0.94796118, 0.88911267,
                            0.92651953, 0.91013375, 0.82531628, 0.65701872};

        auto output = BlstmComputerTest::createAndRun(timesteps, batch, inDim, stateDim, bidirectional, input, weights,
                                                      initH, initC, lengths, Tensor::CAFFE_C4);

        if (!checkVector<float>(output->host<float>(), expected, (bidirectional ? 2 : 1) * stateDim * batch, 0.00001)) {
            MNN_ERROR("BlstmComputerTestNC4HW4 failed!\n");
        }
        return true;
    }
};

MNNTestSuiteRegister(BlstmComputerTestNormal,
        "backend/cpu/compute/blstm_computer_normal");
MNNTestSuiteRegister(BlstmComputerTestUnidirection,
        "backend/cpu/compute/blstm_computer_unidirection");
MNNTestSuiteRegister(BlstmComputerTestNC4HW4,
        "backend/cpu/compute/blstm_computer_nc4hw4");
#endif
