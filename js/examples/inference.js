const mnn = require('../index');
const fs = require('fs');
const path = require('path');

async function run() {
    const modelPath = process.argv[2] || path.join(__dirname, '../test/fixtures/mobilenet.mnn');

    if (!fs.existsSync(modelPath)) {
        console.error('Model not found:', modelPath);
        console.log('Please provide path to model file');
        process.exit(1);
    }

    console.log(`Loading model from ${modelPath}...`);
    const interpreter = mnn.Interpreter.createFromFile(modelPath);
    console.log(`MNN Version: ${mnn.version}`);
    console.log(`Model Version: ${interpreter.getModelVersion()}`);

    // Create session
    const config = {
        type: mnn.ForwardType.CPU,
        numThread: 4
    };
    console.log('Creating session with config:', config);
    const session = interpreter.createSession(config);

    // Get input info
    const input = interpreter.getSessionInput(session);
    const shape = input.getShape();
    console.log('Input Shape:', shape);
    console.log('Input Type:', input.getDataType());

    // Prepare input data
    const size = shape.reduce((a, b) => a * b, 1);
    const inputData = new Float32Array(size);
    console.log(`Filling input with ${size} random values...`);
    for (let i = 0; i < size; i++) {
        inputData[i] = Math.random();
    }
    input.copyFrom(inputData);

    // Run inference
    console.log('Running inference...');
    console.time('Inference');
    const code = interpreter.runSession(session);
    console.timeEnd('Inference');

    if (code !== mnn.ErrorCode.NO_ERROR) {
        console.error('Inference failed with code:', code);
        process.exit(1);
    }

    // Get output
    const output = interpreter.getSessionOutput(session);
    const outputShape = output.getShape();
    const outputData = output.getData();
    console.log('Output Shape:', outputShape);
    console.log('Output Data (first 10):', outputData.subarray(0, 10));

    // Release resources
    interpreter.release();
    console.log('Done');
}

run().catch(console.error);
