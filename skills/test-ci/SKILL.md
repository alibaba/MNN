---
name: test-ci
description: Run MNN tests / benchmarks on host or real devices. Covers two parallel tracks — (1) the regression / CI suite (static checks, host-side tests, on-device Android arm64 matrix via ./test.sh + test_stages.json) and (2) one-command iOS real-device LLM benchmarking (prefill/decode tok/s, branch comparison). Use when the user asks to run the tests, run CI, smoke-test a build, verify a change on a device, benchmark on-device (Android or iPhone/iPad), or add / select / retune a test stage.
---

# MNN Test / CI SKILL (index)

This skill is an **index**. Pick the document that matches the task and follow it:

| Document | Use when |
|----------|----------|
| [`test-suite.md`](test-suite.md) | Run the regression / CI suite — static checks, host (local) tests, the on-device **Android** arm64 matrix (`./test.sh` + `test_stages.json`); add / select / retune a test stage; audit stale CI scripts; add a new operator test. |
| [`ios-llm-bench.md`](ios-llm-bench.md) | Benchmark LLM prefill/decode speed on a real **iPhone/iPad** (`ios_llm_bench.sh`); compare branches on iOS Metal/CPU; verify Metal kernel changes on device. |

The two tracks are independent: Android/host regression testing goes through
`test.sh`, while iOS LLM benchmarking goes through
`transformers/llm/engine/ios/ios_llm_bench.sh`.
