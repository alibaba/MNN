const mnn = require('../index');
const fs = require('fs');
const path = require('path');

const COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

const CONFIG = {
    inputSize: 640,
    confThreshold: 0.25,
    nmsThreshold: 0.45,
    numClasses: 80
};

function preprocessImage(imageData, width, height) {
    const { inputSize } = CONFIG;
    const scale = Math.min(inputSize / width, inputSize / height);
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);
    const padX = (inputSize - newWidth) / 2;
    const padY = (inputSize - newHeight) / 2;

    const inputData = new Float32Array(1 * 3 * inputSize * inputSize);
    inputData.fill(0.5);

    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const srcX = Math.floor(x / scale);
            const srcY = Math.floor(y / scale);
            const srcIdx = (srcY * width + srcX) * 3;
            const dstX = Math.floor(padX) + x;
            const dstY = Math.floor(padY) + y;

            for (let c = 0; c < 3; c++) {
                const dstIdx = c * inputSize * inputSize + dstY * inputSize + dstX;
                inputData[dstIdx] = imageData[srcIdx + c] / 255.0;
            }
        }
    }

    return { inputData, scale, padX, padY };
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

function postprocess(output, scale, padX, padY, origWidth, origHeight) {
    const { confThreshold, nmsThreshold, numClasses, inputSize } = CONFIG;
    const outputData = output.getData();
    const outputShape = output.getShape();

    const numPredictions = outputShape[2];
    const boxes = [];
    const scores = [];
    const classIds = [];

    for (let i = 0; i < numPredictions; i++) {
        const offset = i;
        const stride = numPredictions;

        const cx = outputData[0 * stride + offset];
        const cy = outputData[1 * stride + offset];
        const w = outputData[2 * stride + offset];
        const h = outputData[3 * stride + offset];

        let maxScore = 0;
        let maxClassId = 0;
        for (let c = 0; c < numClasses; c++) {
            const score = outputData[(4 + c) * stride + offset];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore < confThreshold) continue;

        const x1 = ((cx - w / 2) - padX) / scale;
        const y1 = ((cy - h / 2) - padY) / scale;
        const x2 = ((cx + w / 2) - padX) / scale;
        const y2 = ((cy + h / 2) - padY) / scale;

        boxes.push({
            x1: Math.max(0, x1),
            y1: Math.max(0, y1),
            x2: Math.min(origWidth, x2),
            y2: Math.min(origHeight, y2)
        });
        scores.push(maxScore);
        classIds.push(maxClassId);
    }

    const keepIndices = nms(boxes, scores, nmsThreshold);

    return keepIndices.map(i => ({
        box: boxes[i],
        score: scores[i],
        classId: classIds[i],
        className: COCO_CLASSES[classIds[i]]
    }));
}

async function run() {
    const modelPath = process.argv[2] || path.join(__dirname, '../models/yolov8n.mnn');

    if (!fs.existsSync(modelPath)) {
        console.error('Model not found:', modelPath);
        console.log('Usage: node yolo_detection.js <model_path> [image_width] [image_height]');
        console.log('Download YOLOv8 model from: https://github.com/ultralytics/ultralytics');
        process.exit(1);
    }

    const imageWidth = parseInt(process.argv[3]) || 640;
    const imageHeight = parseInt(process.argv[4]) || 480;

    console.log(`Loading YOLO model from ${modelPath}...`);
    const interpreter = mnn.Interpreter.createFromFile(modelPath);
    console.log(`MNN Version: ${mnn.version}`);

    const config = {
        type: mnn.ForwardType.CPU,
        numThread: 4
    };
    console.log('Creating session with config:', config);
    const session = interpreter.createSession(config);

    const input = interpreter.getSessionInput(session);
    const shape = input.getShape();
    console.log('Model Input Shape:', shape);

    console.log(`Generating dummy image data (${imageWidth}x${imageHeight})...`);
    const dummyImage = new Uint8Array(imageWidth * imageHeight * 3);
    for (let i = 0; i < dummyImage.length; i++) {
        dummyImage[i] = Math.floor(Math.random() * 256);
    }

    const { inputData, scale, padX, padY } = preprocessImage(dummyImage, imageWidth, imageHeight);
    input.copyFrom(inputData);

    console.log('Running YOLO inference...');
    console.time('Inference');
    const code = interpreter.runSession(session);
    console.timeEnd('Inference');

    if (code !== mnn.ErrorCode.NO_ERROR) {
        console.error('Inference failed with code:', code);
        process.exit(1);
    }

    const output = interpreter.getSessionOutput(session);
    console.log('Output Shape:', output.getShape());

    const detections = postprocess(output, scale, padX, padY, imageWidth, imageHeight);

    console.log(`\nDetected ${detections.length} objects:`);
    detections.forEach((det, i) => {
        console.log(`  [${i + 1}] ${det.className} (${(det.score * 100).toFixed(1)}%): ` +
            `[${det.box.x1.toFixed(0)}, ${det.box.y1.toFixed(0)}, ` +
            `${det.box.x2.toFixed(0)}, ${det.box.y2.toFixed(0)}]`);
    });

    interpreter.release();
    console.log('\nDone');
}

run().catch(console.error);
