const assert = require('chai').assert;
const fs = require('fs');
const path = require('path');
const MNN = require('..');

describe('MNN Generic LLM', () => {
    it('should have llm module', () => {
        assert.isObject(MNN.llm, 'MNN.llm should be an object');
        assert.isFunction(MNN.llm.create, 'MNN.llm.create should be a function');
    });

    it('should verify llm instance methods', () => {
        // Create a dummy config for API testing
        const configPath = path.join(__dirname, 'dummy_config.json');
        const dummyConfig = {
            "llm_model": "dummy.mnn"
        };
        fs.writeFileSync(configPath, JSON.stringify(dummyConfig));

        try {
            const llm = MNN.llm.create(configPath);
            assert.isObject(llm, 'llm instance created');
            assert.isFunction(llm.load, 'load method exists');
            assert.isFunction(llm.response, 'response method exists');
            assert.isFunction(llm.generate, 'generate method exists');
            assert.isFunction(llm.applyChatTemplate, 'applyChatTemplate method exists');
            assert.isFunction(llm.setConfig, 'setConfig method exists');

            // We don't call load() as it would fail/crash without real model
        } finally {
            if (fs.existsSync(configPath)) {
                fs.unlinkSync(configPath);
            }
        }
    });

    // Conditional test for real inference
    const modelPath = process.env.LLM_MODEL_PATH;
    if (modelPath && fs.existsSync(modelPath)) {
        it('should complete inference with real model', function () {
            this.timeout(60000); // 1 minute timeout for load/inference
            console.log('Testing with model:', modelPath);

            const llm = MNN.llm.create(modelPath);
            const loaded = llm.load();
            assert.isTrue(loaded, 'Model should load successfully');

            const query = 'Hello';
            console.log('Query:', query);
            const response = llm.response(query);
            console.log('Response:', response);
            assert.isString(response);
            assert.isNotEmpty(response);

            const inputIds = [151644, 872, 198]; // Example IDs
            const outputIds = llm.generate(inputIds);
            console.log('Generate output IDs:', outputIds);
            assert.isArray(outputIds);
        });
    } else {
        console.log('Skipping real inference test. Set LLM_MODEL_PATH env var to run it.');
    }
});
