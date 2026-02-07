const assert = require('assert');
const mnn = require('../index');
const path = require('path');
const fs = require('fs');

const MODEL_PATH = path.join(__dirname, 'fixtures', 'mobilenet.mnn');
const hasModel = fs.existsSync(MODEL_PATH);

describe('Tensor', function () {
    let interpreter;
    let session;
    let inputTensor;

    before(function () {
        if (!hasModel) {
            this.skip();
            return;
        }
        interpreter = mnn.Interpreter.createFromFile(MODEL_PATH);
        session = interpreter.createSession();
        inputTensor = interpreter.getSessionInput(session);
    });

    after(function () {
        if (interpreter) {
            interpreter.release();
        }
    });

    it('should have valid dimensions', function () {
        const shape = inputTensor.getShape();
        assert(Array.isArray(shape));
        assert(shape.length > 0);
        // mobilenet usually [1, 3, 224, 224] or similar
        assert.strictEqual(shape.length, 4);
    });

    it('should allow data access and modification', function () {
        const shape = inputTensor.getShape();
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size);
        for(let i=0; i<size; i++) data[i] = 1.0;
        
        // This is a test of the binding's ability to copy data
        inputTensor.copyFrom(data);
        
        // Retrieve and check (if API allows getData)
        const outputData = inputTensor.getData();
        // Note: Floating point comparison might need epsilon, but copyFrom/getData should be exact for 1.0
        // Or at least check length
        assert.strictEqual(outputData.length, size);
        assert(outputData[0] === 1.0 || Math.abs(outputData[0] - 1.0) < 1e-5);
    });

    it('should be an instance of mnn.Tensor', function () {
        assert(inputTensor instanceof mnn.Tensor);
    });
});
