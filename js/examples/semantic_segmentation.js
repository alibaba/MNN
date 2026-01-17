const mnn = require('../index');
const fs = require('fs');
const path = require('path');

const CITYSCAPES_CLASSES = [
    { name: 'road', color: [128, 64, 128] },
    { name: 'sidewalk', color: [244, 35, 232] },
    { name: 'building', color: [70, 70, 70] },
    { name: 'wall', color: [102, 102, 156] },
    { name: 'fence', color: [190, 153, 153] },
    { name: 'pole', color: [153, 153, 153] },
    { name: 'traffic light', color: [250, 170, 30] },
    { name: 'traffic sign', color: [220, 220, 0] },
    { name: 'vegetation', color: [107, 142, 35] },
    { name: 'terrain', color: [152, 251, 152] },
    { name: 'sky', color: [70, 130, 180] },
    { name: 'person', color: [220, 20, 60] },
    { name: 'rider', color: [255, 0, 0] },
    { name: 'car', color: [0, 0, 142] },
    { name: 'truck', color: [0, 0, 70] },
    { name: 'bus', color: [0, 60, 100] },
    { name: 'train', color: [0, 80, 100] },
    { name: 'motorcycle', color: [0, 0, 230] },
    { name: 'bicycle', color: [119, 11, 32] }
];

const ADE20K_CLASSES = [
    { name: 'wall', color: [120, 120, 120] },
    { name: 'floor', color: [180, 120, 120] },
    { name: 'ceiling', color: [6, 230, 230] },
    { name: 'bed', color: [80, 50, 50] },
    { name: 'window', color: [4, 200, 3] },
    { name: 'cabinet', color: [120, 120, 80] },
    { name: 'door', color: [140, 140, 140] },
    { name: 'table', color: [204, 5, 255] },
    { name: 'plant', color: [4, 250, 7] },
    { name: 'chair', color: [224, 5, 255] },
    { name: 'sofa', color: [235, 255, 7] },
    { name: 'lamp', color: [150, 5, 61] },
    { name: 'sky', color: [8, 255, 51] },
    { name: 'person', color: [255, 6, 82] },
    { name: 'car', color: [143, 255, 140] },
    { name: 'water', color: [204, 255, 4] },
    { name: 'grass', color: [255, 51, 7] },
    { name: 'mountain', color: [204, 70, 3] },
    { name: 'tree', color: [0, 102, 200] },
    { name: 'building', color: [61, 230, 250] }
];

const CONFIG = {
    inputWidth: 512,
    inputHeight: 512,
    meanValues: [123.675, 116.28, 103.53],
    stdValues: [58.395, 57.12, 57.375]
};

function preprocessImage(imageData, width, height) {
    const { inputWidth, inputHeight, meanValues, stdValues } = CONFIG;

    const inputData = new Float32Array(1 * 3 * inputHeight * inputWidth);

    for (let y = 0; y < inputHeight; y++) {
        for (let x = 0; x < inputWidth; x++) {
            const srcX = Math.floor(x * width / inputWidth);
            const srcY = Math.floor(y * height / inputHeight);
            const srcIdx = (srcY * width + srcX) * 3;

            for (let c = 0; c < 3; c++) {
                const dstIdx = c * inputHeight * inputWidth + y * inputWidth + x;
                inputData[dstIdx] = (imageData[srcIdx + c] - meanValues[c]) / stdValues[c];
            }
        }
    }

    return inputData;
}

function postprocessOutput(outputData, outputShape, classes) {
    const numClasses = outputShape[1];
    const height = outputShape[2];
    const width = outputShape[3];

    const segmentationMap = new Uint8Array(height * width);
    const classCounts = new Map();

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxProb = -Infinity;
            let maxClass = 0;

            for (let c = 0; c < numClasses; c++) {
                const prob = outputData[c * height * width + y * width + x];
                if (prob > maxProb) {
                    maxProb = prob;
                    maxClass = c;
                }
            }

            segmentationMap[y * width + x] = maxClass;
            classCounts.set(maxClass, (classCounts.get(maxClass) || 0) + 1);
        }
    }

    return { segmentationMap, classCounts, width, height };
}

function generateColorMap(segmentationMap, width, height, classes) {
    const colorMap = new Uint8Array(height * width * 3);

    for (let i = 0; i < segmentationMap.length; i++) {
        const classId = segmentationMap[i];
        const classInfo = classes[classId % classes.length];
        colorMap[i * 3] = classInfo.color[0];
        colorMap[i * 3 + 1] = classInfo.color[1];
        colorMap[i * 3 + 2] = classInfo.color[2];
    }

    return colorMap;
}

async function run() {
    const modelPath = process.argv[2] || path.join(__dirname, '../models/deeplabv3.mnn');
    const datasetType = process.argv[3] || 'cityscapes';

    if (!fs.existsSync(modelPath)) {
        console.error('Model not found:', modelPath);
        console.log('\nUsage: node semantic_segmentation.js <model_path> [dataset_type] [image_width] [image_height]');
        console.log('\nDataset types: cityscapes, ade20k');
        console.log('\nSupported models:');
        console.log('  - DeepLabV3/V3+');
        console.log('  - PSPNet');
        console.log('  - FCN');
        console.log('  - UNet');
        console.log('  - SegFormer');
        console.log('\nModels can be obtained from:');
        console.log('  - https://github.com/tensorflow/models/tree/master/research/deeplab');
        console.log('  - https://github.com/NVlabs/SegFormer');
        process.exit(1);
    }

    const classes = datasetType === 'ade20k' ? ADE20K_CLASSES : CITYSCAPES_CLASSES;
    const imageWidth = parseInt(process.argv[4]) || 640;
    const imageHeight = parseInt(process.argv[5]) || 480;

    console.log(`Loading segmentation model from ${modelPath}...`);
    console.log(`Dataset: ${datasetType} (${classes.length} classes)`);
    const interpreter = mnn.Interpreter.createFromFile(modelPath);
    console.log(`MNN Version: ${mnn.version}`);

    const config = {
        type: mnn.ForwardType.CPU,
        numThread: 4
    };
    console.log('Creating session with config:', config);
    const session = interpreter.createSession(config);

    const input = interpreter.getSessionInput(session);
    const inputShape = input.getShape();
    console.log('Model Input Shape:', inputShape);

    console.log(`\nGenerating dummy image (${imageWidth}x${imageHeight})...`);
    const dummyImage = new Uint8Array(imageWidth * imageHeight * 3);
    for (let i = 0; i < dummyImage.length; i++) {
        dummyImage[i] = Math.floor(Math.random() * 256);
    }

    const inputData = preprocessImage(dummyImage, imageWidth, imageHeight);
    input.copyFrom(inputData);

    console.log('Running semantic segmentation...');
    console.time('Segmentation');
    const code = interpreter.runSession(session);
    console.timeEnd('Segmentation');

    if (code !== mnn.ErrorCode.NO_ERROR) {
        console.error('Inference failed with code:', code);
        process.exit(1);
    }

    const output = interpreter.getSessionOutput(session);
    const outputShape = output.getShape();
    const outputData = output.getData();
    console.log('Output Shape:', outputShape);

    const { segmentationMap, classCounts, width, height } = postprocessOutput(
        outputData, outputShape, classes
    );

    console.log(`\nSegmentation map size: ${width}x${height}`);
    console.log('\nClass distribution:');

    const totalPixels = width * height;
    const sortedCounts = Array.from(classCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

    sortedCounts.forEach(([classId, count]) => {
        const classInfo = classes[classId % classes.length];
        const percentage = (count / totalPixels * 100).toFixed(2);
        console.log(`  ${classInfo.name}: ${percentage}% (${count} pixels)`);
    });

    const colorMap = generateColorMap(segmentationMap, width, height, classes);
    console.log(`\nGenerated color-coded segmentation map (${width}x${height}x3)`);

    interpreter.release();
    console.log('\nSemantic segmentation complete!');
    console.log('In a real application, you would save the segmentation map to a file.');
}

run().catch(console.error);
