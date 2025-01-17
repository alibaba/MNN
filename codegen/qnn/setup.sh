# set environment var
export MNN_ROOT=$(pwd)
echo "MNN_ROOT: ${MNN_ROOT}"
export QNN_LIB_ROOT=${MNN_ROOT}/source/backend/qnn/3rd_party/
echo "QNN_LIB_ROOT: ${QNN_LIB_ROOT}"

# generate json config file
echo "{\n\t\"CustomInstallPath\" : \"${QNN_LIB_ROOT}/Hexagon\"\n}" > ${MNN_ROOT}/codegen/qnn/install_Hexagon.json
echo "{\n\t\"CustomInstallPath\" : \"${QNN_LIB_ROOT}/qnn_ai\"\n}" > ${MNN_ROOT}/codegen/qnn/install_qnn_ai.json

# download hexagonsdk5
if (test -d ${QNN_LIB_ROOT}/Hexagon) then
    echo "Hexagon SDK is ready!"
else
    echo "Start downloading Hexagon SDK..."
    qpm-cli --license-activate hexagonsdk5.x
    export HEXAGONSDK_PATH=$(qpm-cli --download-only hexagonsdk5.x | grep -oP '(?<=\[Info\] : Downloaded file : ).*')
    echo "Downloaded at ${HEXAGONSDK_PATH}"
    qpm-cli --extract ${HEXAGONSDK_PATH} --config ${MNN_ROOT}/codegen/qnn/install_Hexagon.json
    rm ${HEXAGONSDK_PATH}
    echo "Hexagon SDK is ready!"
fi

# download qualcomm_neural_processing_sdk
if (test -d ${QNN_LIB_ROOT}/qnn_ai) then
    echo "qualcomm_neural_processing_sdk is ready!"
else
    echo "Start downloading qualcomm_neural_processing_sdk..."
    qpm-cli --license-activate qualcomm_neural_processing_sdk
    export QNN_AI_PATH=$(qpm-cli --download-only qualcomm_neural_processing_sdk | grep -oP '(?<=\[Info\] : Downloaded file : ).*')
    echo "Downloaded at ${QNN_AI_PATH}"
    qpm-cli --extract ${QNN_AI_PATH} --config ${MNN_ROOT}/codegen/qnn/install_qnn_ai.json 
    rm ${QNN_AI_PATH}
    echo "qualcomm_neural_processing_sdk is ready!"
fi