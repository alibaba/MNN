
const mnn = require('./index.js');
const fs = require('fs');
const { Jimp } = require('jimp');
const path = require('path');

async function runStyleTransfer() {
    const modelPath = path.join(__dirname, 'test_assets/style_transfer.mnn');
    const inputImagePath = path.join(__dirname, 'test_assets/input.png');
    const outputImagePath = path.join(__dirname, 'test_assets/output.png');

    console.log(`Loading model from ${modelPath}...`);
    if (!fs.existsSync(modelPath)) {
        console.error('Model file not found!');
        process.exit(1);
    }

    // 1. Create Interpreter and Session
    const interpreter = mnn.Interpreter.createFromFile(modelPath);
    const session = interpreter.createSession({
        type: mnn.ForwardType.CPU,
        numThread: 4
    });

    // 2. Get Input Tensor and check shape
    const inputTensor = interpreter.getSessionInput(session, 'input1');
    const shape = inputTensor.getShape(); // Expected: [1, 3, 224, 224]
    console.log('Input shape:', shape);

    const N = shape[0];
    const C = shape[1];
    const H = shape[2];
    const W = shape[3];

    // 3. Load and Preprocess Image
    console.log(`Loading image from ${inputImagePath}...`);
    const image = await Jimp.read(inputImagePath);
    image.resize({ w: W, h: H });

    // Prepare input data (NCHW float32)
    const inputData = new Float32Array(N * C * H * W);

    const { data, width, height } = image.bitmap;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (width * y + x) << 2;
            const r = data[idx + 0];
            const g = data[idx + 1];
            const b = data[idx + 2];

            // NCHW layout
            inputData[0 * H * W + y * W + x] = r;
            inputData[1 * H * W + y * W + x] = g;
            inputData[2 * H * W + y * W + x] = b;
        }
    }

    inputTensor.copyFrom(inputData);

    // 4. Run Inference
    console.log('Running inference...');
    interpreter.runSession(session);

    // 5. Get Output
    const outputTensor = interpreter.getSessionOutput(session, 'output1');
    const outputData = outputTensor.getData(); // Float32Array
    const outShape = outputTensor.getShape();
    console.log('Output shape:', outShape);

    // 6. Postprocess and Save
    const outH = outShape[2];
    const outW = outShape[3];
    const outputImage = new Jimp({ width: outW, height: outH });

    // Need to access bitmap data directly
    const outBitmapData = outputImage.bitmap.data;

    for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
            // NCHW -> HWC
            let r = outputData[0 * outH * outW + y * outW + x];
            let g = outputData[1 * outH * outW + y * outW + x];
            let b = outputData[2 * outH * outW + y * outW + x];

            // Clamp values
            r = Math.max(0, Math.min(255, r));
            g = Math.max(0, Math.min(255, g));
            b = Math.max(0, Math.min(255, b));

            const idx = (outW * y + x) << 2;
            outBitmapData[idx + 0] = r;
            outBitmapData[idx + 1] = g;
            outBitmapData[idx + 2] = b;
            outBitmapData[idx + 3] = 255; // Alpha
        }
    }

    console.log(`Saving output to ${outputImagePath}...`);
    await outputImage.write(outputImagePath);
    console.log('Done!');

    // Cleanup
    interpreter.release();
}

runStyleTransfer().catch(console.error);
