const mnn = require('../index');
const fs = require('fs');
const path = require('path');

const STYLE_NAMES = [
    'candy',
    'mosaic',
    'rain_princess',
    'udnie',
    'starry_night',
    'la_muse',
    'wave',
    'scream'
];

const CONFIG = {
    inputSize: 224,
    meanValues: [123.68, 116.779, 103.939],
    normValues: [1.0, 1.0, 1.0]
};

function preprocessImage(imageData, width, height) {
    const { inputSize, meanValues, normValues } = CONFIG;

    const scale = Math.min(inputSize / width, inputSize / height);
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);
    const padX = Math.floor((inputSize - newWidth) / 2);
    const padY = Math.floor((inputSize - newHeight) / 2);

    const inputData = new Float32Array(1 * 3 * inputSize * inputSize);
    inputData.fill(0);

    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const srcX = Math.min(Math.floor(x / scale), width - 1);
            const srcY = Math.min(Math.floor(y / scale), height - 1);
            const srcIdx = (srcY * width + srcX) * 3;
            const dstX = padX + x;
            const dstY = padY + y;

            for (let c = 0; c < 3; c++) {
                const dstIdx = c * inputSize * inputSize + dstY * inputSize + dstX;
                inputData[dstIdx] = (imageData[srcIdx + c] - meanValues[c]) * normValues[c];
            }
        }
    }

    return { inputData, scale, padX, padY, newWidth, newHeight };
}

function postprocessOutput(outputData, outputShape, padX, padY, newWidth, newHeight) {
    const h = outputShape[2];
    const w = outputShape[3];

    const result = new Uint8Array(newWidth * newHeight * 3);

    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const srcX = padX + x;
            const srcY = padY + y;
            const dstIdx = (y * newWidth + x) * 3;

            for (let c = 0; c < 3; c++) {
                const srcIdx = c * h * w + srcY * w + srcX;
                let value = outputData[srcIdx];
                value = Math.max(0, Math.min(255, Math.round(value)));
                result[dstIdx + c] = value;
            }
        }
    }

    return result;
}

async function run() {
    const modelPath = process.argv[2] || path.join(__dirname, '../models/style_transfer.mnn');
    const styleName = process.argv[3] || 'candy';

    if (!fs.existsSync(modelPath)) {
        console.error('Model not found:', modelPath);
        console.log('\nUsage: node style_transfer.js <model_path> [style_name]');
        console.log('\nAvailable styles:', STYLE_NAMES.join(', '));
        console.log('\nStyle transfer models can be obtained from:');
        console.log('  - https://github.com/onnx/models/tree/main/validated/vision/style_transfer');
        console.log('  - https://github.com/pytorch/examples/tree/main/fast_neural_style');
        process.exit(1);
    }

    const imageWidth = parseInt(process.argv[4]) || 224;
    const imageHeight = parseInt(process.argv[5]) || 224;

    console.log(`Loading Style Transfer model from ${modelPath}...`);
    console.log(`Style: ${styleName}`);
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

    console.log(`\nGenerating dummy content image (${imageWidth}x${imageHeight})...`);
    const dummyImage = new Uint8Array(imageWidth * imageHeight * 3);
    for (let i = 0; i < dummyImage.length; i++) {
        dummyImage[i] = Math.floor(Math.random() * 256);
    }

    const { inputData, scale, padX, padY, newWidth, newHeight } = preprocessImage(
        dummyImage, imageWidth, imageHeight
    );
    input.copyFrom(inputData);

    console.log('Running style transfer inference...');
    console.time('Style Transfer');
    const code = interpreter.runSession(session);
    console.timeEnd('Style Transfer');

    if (code !== mnn.ErrorCode.NO_ERROR) {
        console.error('Inference failed with code:', code);
        process.exit(1);
    }

    const output = interpreter.getSessionOutput(session);
    const outputShape = output.getShape();
    const outputData = output.getData();
    console.log('Output Shape:', outputShape);

    const styledImage = postprocessOutput(outputData, outputShape, padX, padY, newWidth, newHeight);
    console.log(`Styled image size: ${newWidth}x${newHeight}`);

    let avgR = 0, avgG = 0, avgB = 0;
    const numPixels = newWidth * newHeight;
    for (let i = 0; i < numPixels; i++) {
        avgR += styledImage[i * 3];
        avgG += styledImage[i * 3 + 1];
        avgB += styledImage[i * 3 + 2];
    }
    console.log(`Average output color (RGB): [${(avgR/numPixels).toFixed(1)}, ` +
        `${(avgG/numPixels).toFixed(1)}, ${(avgB/numPixels).toFixed(1)}]`);

    interpreter.release();
    console.log('\nStyle transfer complete!');
    console.log('In a real application, you would save the styled image to a file.');
}

run().catch(console.error);
