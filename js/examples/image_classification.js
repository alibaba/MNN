const mnn = require('../index');
const fs = require('fs');
const path = require('path');

const IMAGENET_CLASSES = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
    'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin',
    'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite',
    'bald eagle', 'vulture', 'great grey owl', 'fire salamander',
    'smooth newt', 'newt', 'spotted salamander', 'axolotl', 'bullfrog',
    'tree frog', 'tailed frog', 'loggerhead', 'leatherback turtle',
    'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana',
    'American chameleon', 'whiptail', 'agama', 'frilled lizard',
    'alligator lizard', 'Gila monster', 'green lizard', 'African chameleon',
    'Komodo dragon', 'African crocodile', 'American alligator',
    'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake',
    'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake',
    'night snake', 'boa constrictor', 'rock python', 'Indian cobra',
    'green mamba', 'sea snake', 'horned viper', 'diamondback',
    'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'black and gold garden spider',
    'barn spider', 'garden spider', 'black widow', 'tarantula', 'wolf spider',
    'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse',
    'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey',
    'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater',
    'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser',
    'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby',
    'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm',
    'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
    'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab',
    'king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab',
    'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo',
    'little blue heron', 'American egret', 'bittern', 'crane', 'limpkin',
    'European gallinule', 'American coot', 'bustard', 'ruddy turnstone',
    'red-backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican',
    'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong',
    'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese',
    'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier',
    'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle', 'bloodhound',
    'bluetick', 'black-and-tan coonhound', 'Walker hound', 'English foxhound',
    'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet'
];

const CONFIG = {
    inputSize: 224,
    meanValues: [0.485, 0.456, 0.406],
    stdValues: [0.229, 0.224, 0.225],
    topK: 5
};

function preprocessImage(imageData, width, height) {
    const { inputSize, meanValues, stdValues } = CONFIG;

    const inputData = new Float32Array(1 * 3 * inputSize * inputSize);

    for (let y = 0; y < inputSize; y++) {
        for (let x = 0; x < inputSize; x++) {
            const srcX = Math.floor(x * width / inputSize);
            const srcY = Math.floor(y * height / inputSize);
            const srcIdx = (srcY * width + srcX) * 3;

            for (let c = 0; c < 3; c++) {
                const dstIdx = c * inputSize * inputSize + y * inputSize + x;
                const value = imageData[srcIdx + c] / 255.0;
                inputData[dstIdx] = (value - meanValues[c]) / stdValues[c];
            }
        }
    }

    return inputData;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(e => e / sumExp);
}

function getTopK(probs, k) {
    const indexed = probs.map((p, i) => ({ prob: p, index: i }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
}

async function run() {
    const modelPath = process.argv[2] || path.join(__dirname, '../models/mobilenet.mnn');

    if (!fs.existsSync(modelPath)) {
        console.error('Model not found:', modelPath);
        console.log('\nUsage: node image_classification.js <model_path> [image_width] [image_height]');
        console.log('\nSupported models:');
        console.log('  - MobileNet V1/V2/V3');
        console.log('  - ResNet-18/34/50/101/152');
        console.log('  - EfficientNet-B0 to B7');
        console.log('  - VGG-16/19');
        console.log('\nModels can be converted to MNN format using MNNConvert tool.');
        process.exit(1);
    }

    const imageWidth = parseInt(process.argv[3]) || 224;
    const imageHeight = parseInt(process.argv[4]) || 224;

    console.log(`Loading classification model from ${modelPath}...`);
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

    console.log('Running classification inference...');
    console.time('Classification');
    const code = interpreter.runSession(session);
    console.timeEnd('Classification');

    if (code !== mnn.ErrorCode.NO_ERROR) {
        console.error('Inference failed with code:', code);
        process.exit(1);
    }

    const output = interpreter.getSessionOutput(session);
    const outputShape = output.getShape();
    const outputData = output.getData();
    console.log('Output Shape:', outputShape);

    const probs = softmax(Array.from(outputData));
    const topK = getTopK(probs, CONFIG.topK);

    console.log(`\nTop-${CONFIG.topK} Predictions:`);
    topK.forEach((item, rank) => {
        const className = IMAGENET_CLASSES[item.index] || `class_${item.index}`;
        console.log(`  ${rank + 1}. ${className}: ${(item.prob * 100).toFixed(2)}%`);
    });

    interpreter.release();
    console.log('\nClassification complete!');
}

run().catch(console.error);
