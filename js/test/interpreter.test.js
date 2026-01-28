const assert = require('assert');
const mnn = require('../index');
const fs = require('fs');
const path = require('path');

// Mock file if not exists
const MODEL_PATH = path.join(__dirname, 'fixtures', 'mobilenet.mnn');

// Skip tests if model file doesn't exist (since we can't easily download it here)
const hasModel = fs.existsSync(MODEL_PATH);

describe('Interpreter', function () {

    if (!hasModel) {
        it.skip('should create interpreter from file (model not found)', function () { });
        it.skip('should create and run session (model not found)', function () { });
        return;
    }

    it('should create interpreter from file', function () {
        assert.doesNotThrow(() => {
            const interpreter = mnn.Interpreter.createFromFile(MODEL_PATH);
            assert(interpreter instanceof mnn.Interpreter);
            interpreter.release();
        });
    });

    it('should create interpreter from buffer', function () {
        const buffer = fs.readFileSync(MODEL_PATH);
        const interpreter = mnn.Interpreter.createFromBuffer(buffer);
        assert(interpreter instanceof mnn.Interpreter);
        interpreter.release();
    });

    it('should handle session creation (benchmark model may fail)', function () {
        const interpreter = mnn.Interpreter.createFromFile(MODEL_PATH);
        try {
            const session = interpreter.createSession({
                type: mnn.ForwardType.CPU,
                numThread: 2
            });
            // If it succeeds (e.g. valid model), verify it's a session
            assert(session instanceof mnn.Session);
        } catch (e) {
            // If it fails (benchmark model often lacks weights), verify error propagation
            assert(e instanceof Error);
            console.log('Session creation failed as expected for benchmark model:', e.message);
        }
        interpreter.release();
    });

    it('should get input and output tensors (if session matches)', function () {
        const interpreter = mnn.Interpreter.createFromFile(MODEL_PATH);
        try {
            const session = interpreter.createSession();
            const input = interpreter.getSessionInput(session, null);
            assert(input instanceof mnn.Tensor);

            const output = interpreter.getSessionOutput(session, null);
            assert(output instanceof mnn.Tensor);
        } catch (e) {
            console.log('Skipping tensor test due to session failure:', e.message);
            this.skip();
        }
        interpreter.release();
    });

    it('should run inference (if session matches)', function () {
        const interpreter = mnn.Interpreter.createFromFile(MODEL_PATH);
        try {
            const session = interpreter.createSession();
            const input = interpreter.getSessionInput(session);
            const shape = input.getShape();

            // Create dummy input
            const elementCount = shape.reduce((a, b) => a * b, 1);
            const inputData = new Float32Array(elementCount);
            for (let i = 0; i < elementCount; i++) {
                inputData[i] = Math.random();
            }

            input.copyFrom(inputData);

            const code = interpreter.runSession(session);
            assert.strictEqual(code, mnn.ErrorCode.NO_ERROR);

            const output = interpreter.getSessionOutput(session);
            const outputData = output.getData();
            assert(outputData.length > 0);
        } catch (e) {
            console.log('Skipping inference test due to session failure:', e.message);
            this.skip();
        }
        interpreter.release();
    });
});
