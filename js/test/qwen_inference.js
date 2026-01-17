const MNN = require('..');
const path = require('path');

console.log('MNN version:', MNN.version);
console.log('Testing MNN.llm with Qwen3-0.6B model...\n');

const modelPath = path.join(__dirname, '../../models/qwen3-0.6b/config.json');
console.log('Model config path:', modelPath);

try {
    // Create LLM instance
    console.log('\n1. Creating LLM instance...');
    const llm = MNN.llm.create(modelPath);
    console.log('✓ LLM instance created');

    // Load model
    console.log('\n2. Loading model...');
    const startLoad = Date.now();
    const loaded = llm.load();
    const loadTime = Date.now() - startLoad;
    console.log(`✓ Model loaded in ${loadTime}ms`);

    if (!loaded) {
        console.error('✗ Failed to load model');
        process.exit(1);
    }

    // Test 1: Simple response
    console.log('\n3. Testing response() method...');
    const query1 = '你好';
    console.log(`Query: "${query1}"`);
    const startResponse1 = Date.now();
    const response1 = llm.response(query1);
    const responseTime1 = Date.now() - startResponse1;
    console.log(`Response: "${response1}"`);
    console.log(`Time: ${responseTime1}ms`);

    // Test 2: Response without history
    console.log('\n4. Testing response() with history=false...');
    const query2 = '介绍一下你自己';
    console.log(`Query: "${query2}"`);
    const startResponse2 = Date.now();
    const response2 = llm.response(query2, false);
    const responseTime2 = Date.now() - startResponse2;
    console.log(`Response: "${response2}"`);
    console.log(`Time: ${responseTime2}ms`);

    // Test 3: Generate with token IDs
    console.log('\n5. Testing generate() method...');
    // Example token IDs for Qwen (these are just examples, actual IDs depend on tokenizer)
    const inputIds = [151644, 872, 198, 108386, 151645];
    console.log(`Input IDs: [${inputIds.join(', ')}]`);
    const startGenerate = Date.now();
    const outputIds = llm.generate(inputIds);
    const generateTime = Date.now() - startGenerate;
    console.log(`Output IDs: [${outputIds.slice(0, 20).join(', ')}${outputIds.length > 20 ? '...' : ''}]`);
    console.log(`Generated ${outputIds.length} tokens in ${generateTime}ms`);

    console.log('\n✓ All tests passed!');
} catch (error) {
    console.error('\n✗ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
}
