const mnn = require('../index');
const fs = require('fs');
const path = require('path');

const CONFIG = {
    inputSize: 320,
    confThreshold: 0.7,
    nmsThreshold: 0.3,
    varianceX: 0.1,
    varianceY: 0.2,
    minSizes: [[16, 32], [64, 128], [256, 512]],
    steps: [8, 16, 32]
};

function generateAnchors(width, height) {
    const { minSizes, steps, inputSize } = CONFIG;
    const anchors = [];

    for (let k = 0; k < minSizes.length; k++) {
        const step = steps[k];
        const featureWidth = Math.ceil(inputSize / step);
        const featureHeight = Math.ceil(inputSize / step);

        for (let i = 0; i < featureHeight; i++) {
            for (let j = 0; j < featureWidth; j++) {
                for (const minSize of minSizes[k]) {
                    const cx = (j + 0.5) * step / inputSize;
                    const cy = (i + 0.5) * step / inputSize;
                    const s = minSize / inputSize;
                    anchors.push({ cx, cy, w: s, h: s });
                }
            }
        }
    }

    return anchors;
}

function preprocessImage(imageData, width, height) {
    const { inputSize } = CONFIG;

    const inputData = new Float32Array(1 * 3 * inputSize * inputSize);

    for (let y = 0; y < inputSize; y++) {
        for (let x = 0; x < inputSize; x++) {
            const srcX = Math.floor(x * width / inputSize);
            const srcY = Math.floor(y * height / inputSize);
            const srcIdx = (srcY * width + srcX) * 3;

            for (let c = 0; c < 3; c++) {
                const dstIdx = c * inputSize * inputSize + y * inputSize + x;
                inputData[dstIdx] = imageData[srcIdx + c] - 127.5;
            }
        }
    }

    return inputData;
}

function nms(boxes, scores, threshold) {
    const indices = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
    const keep = [];

    while (indices.length > 0) {
        const i = indices.shift();
        keep.push(i);

        const remaining = [];
        for (const j of indices) {
            const iou = computeIoU(boxes[i], boxes[j]);
            if (iou < threshold) {
                remaining.push(j);
            }
        }
        indices.length = 0;
        indices.push(...remaining);
    }
    return keep;
}

function computeIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return intersection / union;
}

function decodeBoxes(locData, confData, anchors, origWidth, origHeight) {
    const { confThreshold, nmsThreshold, varianceX, varianceY, inputSize } = CONFIG;
    const boxes = [];
    const scores = [];
    const landmarks = [];

    const numAnchors = anchors.length;

    for (let i = 0; i < numAnchors; i++) {
        const conf = confData[i * 2 + 1];
        if (conf < confThreshold) continue;

        const anchor = anchors[i];

        const dx = locData[i * 4];
        const dy = locData[i * 4 + 1];
        const dw = locData[i * 4 + 2];
        const dh = locData[i * 4 + 3];

        const cx = anchor.cx + dx * varianceX * anchor.w;
        const cy = anchor.cy + dy * varianceX * anchor.h;
        const w = anchor.w * Math.exp(dw * varianceY);
        const h = anchor.h * Math.exp(dh * varianceY);

        const x1 = (cx - w / 2) * origWidth;
        const y1 = (cy - h / 2) * origHeight;
        const x2 = (cx + w / 2) * origWidth;
        const y2 = (cy + h / 2) * origHeight;

        boxes.push({
            x1: Math.max(0, x1),
            y1: Math.max(0, y1),
            x2: Math.min(origWidth, x2),
            y2: Math.min(origHeight, y2)
        });
        scores.push(conf);
    }

    const keepIndices = nms(boxes, scores, nmsThreshold);

    return keepIndices.map(i => ({
        box: boxes[i],
        score: scores[i]
    }));
}

async function run() {
    const modelPath = process.argv[2] || path.join(__dirname, '../models/retinaface.mnn');

    if (!fs.existsSync(modelPath)) {
        console.error('Model not found:', modelPath);
        console.log('\nUsage: node face_detection.js <model_path> [image_width] [image_height]');
        console.log('\nSupported models:');
        console.log('  - RetinaFace');
        console.log('  - MTCNN');
        console.log('  - BlazeFace');
        console.log('  - UltraFace');
        console.log('\nModels can be obtained from:');
        console.log('  - https://github.com/biubug6/Pytorch_Retinaface');
        console.log('  - https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB');
        process.exit(1);
    }

    const imageWidth = parseInt(process.argv[3]) || 640;
    const imageHeight = parseInt(process.argv[4]) || 480;

    console.log(`Loading face detection model from ${modelPath}...`);
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
    for (let y = 0; y < imageHeight; y++) {
        for (let x = 0; x < imageWidth; x++) {
            const idx = (y * imageWidth + x) * 3;
            dummyImage[idx] = 128 + Math.floor(Math.random() * 64);
            dummyImage[idx + 1] = 100 + Math.floor(Math.random() * 64);
            dummyImage[idx + 2] = 80 + Math.floor(Math.random() * 64);
        }
    }

    const anchors = generateAnchors(imageWidth, imageHeight);
    console.log(`Generated ${anchors.length} anchor boxes`);

    const inputData = preprocessImage(dummyImage, imageWidth, imageHeight);
    input.copyFrom(inputData);

    console.log('Running face detection...');
    console.time('Face Detection');
    const code = interpreter.runSession(session);
    console.timeEnd('Face Detection');

    if (code !== mnn.ErrorCode.NO_ERROR) {
        console.error('Inference failed with code:', code);
        process.exit(1);
    }

    const output = interpreter.getSessionOutput(session);
    const outputShape = output.getShape();
    const outputData = output.getData();
    console.log('Output Shape:', outputShape);

    const numElements = outputData.length;
    const halfLen = Math.floor(numElements / 2);
    const locData = outputData.subarray(0, halfLen);
    const confData = outputData.subarray(halfLen);

    const faces = decodeBoxes(locData, confData, anchors, imageWidth, imageHeight);

    console.log(`\nDetected ${faces.length} faces:`);
    faces.forEach((face, i) => {
        const { box, score } = face;
        const width = box.x2 - box.x1;
        const height = box.y2 - box.y1;
        console.log(`  [${i + 1}] Confidence: ${(score * 100).toFixed(1)}%`);
        console.log(`      Position: [${box.x1.toFixed(0)}, ${box.y1.toFixed(0)}]`);
        console.log(`      Size: ${width.toFixed(0)}x${height.toFixed(0)}`);
    });

    interpreter.release();
    console.log('\nFace detection complete!');
}

run().catch(console.error);
