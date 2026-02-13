const MNN = require('../index');
const path = require('path');
const fs = require('fs');

console.log('MNN version:', MNN.version);

const modelPath = process.argv[2] || path.join(__dirname, '../../models/qwen3-0.6b/config.json');

if (!fs.existsSync(modelPath)) {
    console.error('Model config not found:', modelPath);
    console.log('\nUsage: node qwen_inference.js <config_path>');
    process.exit(1);
}

console.log('Loading model from:', modelPath);

try {
    // Create LLM instance
    console.log('Creating LLM instance...');
    const llm = MNN.llm.create(modelPath);

    // Load model
    console.log('Loading model...');
    const startLoad = Date.now();
    const loaded = llm.load();
    const loadTime = Date.now() - startLoad;
    console.log(`Model loaded in ${loadTime}ms`);

    if (!loaded) {
        console.error('Failed to load model');
        process.exit(1);
    }

    // Example 1: Simple response
    console.log('\n--- Example 1: Simple Chat ---');
    const query1 = '你好';
    console.log(`User: ${query1}`);
    const startResponse1 = Date.now();
    const response1 = llm.response(query1);
    const responseTime1 = Date.now() - startResponse1;
    console.log(`Assistant: ${response1}`);
    console.log(`(Time: ${responseTime1}ms)`);

    // Example 2: Response without history
    console.log('\n--- Example 2: Chat without History ---');
    const query2 = '介绍一下你自己';
    console.log(`User: ${query2}`);
    const startResponse2 = Date.now();
    const response2 = llm.response(query2, false);
    const responseTime2 = Date.now() - startResponse2;
    console.log(`Assistant: ${response2}`);
    console.log(`(Time: ${responseTime2}ms)`);

    // Example 3: Token Generation
    console.log('\n--- Example 3: Token Generation ---');
    // Example token IDs for Qwen (actual IDs depend on tokenizer)
    const inputIds = [151644, 872, 198, 108386, 151645];
    console.log(`Input IDs: [${inputIds.join(', ')}]`);
    const startGenerate = Date.now();
    const outputIds = llm.generate(inputIds);
    const generateTime = Date.now() - startGenerate;
    console.log(`Output IDs: [${outputIds.slice(0, 20).join(', ')}${outputIds.length > 20 ? '...' : ''}]`);
    console.log(`Generated ${outputIds.length} tokens in ${generateTime}ms`);

} catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
}
