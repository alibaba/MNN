---
name: hexagon-optimization
description: Optimize, refactor, build, and regression-test MNN Hexagon/HVX/HMX DSP code paths under `source/backend/hexagon`, using device profile data, correctness checks, and cDSP crash diagnostics.
---

# Hexagon Optimization

Use this skill for MNN Hexagon DSP work where correctness, device stability, and measured `DSPOpType` profile data matter more than speculative micro-optimizations.

## Hard Constraints

- Do not use CPU fallback as an optimization. Do not reject Hexagon support, route selected ops to CPU, or keep changes that improve timing by moving work off Hexagon.
- Do not use operator fusion as an optimization unless the user explicitly lifts this restriction. This includes graph-level fusion, host command fusion, DSP command-group sequence fusion, and replacing multiple ops with a fused custom op.
- Keep optimizations in Hexagon/HTP/DSP scheduling, kernels, tiling, memory movement, resource locking, profiling cleanup, or other backend-preserving implementation details.

## Workflow

1. Inspect the existing implementation before editing:
   - Host/backend code under `source/backend/hexagon`.
   - DSP kernels and dispatch under `source/backend/hexagon/htp-ops-lib/src/dsp`.
   - Reuse local helpers and existing HVX/HMX/DMA utilities instead of adding parallel abstractions.
2. Keep changes scoped and behavior-preserving first:
   - Prefer small mechanical refactors before changing algorithms.
   - Avoid increasing worker stack size unless explicitly requested.
   - Do not leave temporary profiling, debug logs, or experimental guards in final code.
3. Validate every meaningful DSP change:
   - Rebuild DSP `.so`.
   - Copy generated `.so` files into `project/android/build_64`.
   - Run the relevant model test and inspect `DSPOpType` profile output.
   - Treat cDSP/qurt crashes, device disconnects, and profile RPC failures as test failures.
4. Compare the target op directly:
   - Use `DSPOpType <OP>` profile numbers, not only RTF or wall time.
   - Run more than once before judging small gains.
   - Keep only changes that improve the target op without hurting correctness or stability.

## Build And Sync

Run commands from the directories expected by the project scripts.

- Android build directory:
  - `cd project/android/build_64`
- Rebuild host Android artifacts and update device:
  - `cd project/android/build_64 && ../build_64.sh -DMNN_HEXAGON=ON -DMNN_GPU_TIME_PROFILE=ON && ../updateTest.sh`
  - `updateTest.sh` may print `adb: error: cannot stat` for optional binaries that are not built. Verify required binaries such as `libMNN.so`, `ModuleBasic.out`, and the target demo were pushed.
- Rebuild DSP-side code after editing `source/backend/hexagon/htp-ops-lib/src/dsp/*`:
  - `cd source/backend/hexagon/htp-ops-lib && source ~/.bash_profile && sh sync_remote_build.sh`
  - For a specific architecture, pass it explicitly, for example `sh sync_remote_build.sh v79`.
  - Confirm the script reports no unexpected undefined symbols and pushes the rebuilt DSP libraries.
- Copy rebuilt DSP libraries into the Android build directory when testing locally:
  - `cp source/backend/hexagon/htp-ops-lib/outputs/libMNN_htpops.so project/android/build_64/libMNN_htpops.so`
  - `cp source/backend/hexagon/htp-ops-lib/outputs/libMNN_htpops_skel.so project/android/build_64/libMNN_htpops_skel.so`

## Profile And Crash Checks

- Clear logs before a run when checking failures:
  - `adb logcat -c`
- Pull logs after a run:
  - `adb logcat -d`
- Search for failures:
  - `rg -i "execute_command_group_profile failed|qurt|sysfatal|fatal|crash|tlb|cdsp.*crash|adsp.*crash|segv|signal 11"`
- If a run fails, times out, returns a profile RPC error, or the device briefly
  disconnects, also inspect cDSP tombstone/ramdump locations before continuing:
  - `adb shell "ls -lt /data/tombstones /data/vendor/tombstones /data/vendor/ramdump /data/vendor/ssrdump 2>/dev/null | head -80"`
  - `adb shell "find /data/tombstones /data/vendor/tombstones /data/vendor/ramdump /data/vendor/ssrdump -maxdepth 2 -type f 2>/dev/null | tail -40"`
  - Pull only the newest relevant file or small directory to a temporary local path for inspection.
  - Search pulled files for DSP failure signatures:
    - `rg -i "cdsp|adsp|qurt|sysfatal|fatal|crash|tlb|page fault|protection|signal|MNN|htp|fastrpc" <pulled_path>`
  - Remove temporary pulled tombstone/ramdump files after summarizing the useful signal.
- Treat these as failures:
  - `[Hexagon] execute_command_group_profile failed with code -2147482610`
  - qurt/cDSP fatal logs.
  - New or updated cDSP tombstone/ramdump files from the test window.
  - Device offline/drop during the test.
  - Missing or stale rebuilt `.so` after DSP edits.
- Useful profile fields:
  - `Hexagon DSP Profile`
  - `Command groups`
  - `Command dirty`
  - `DSPOpType <name> (<id>): <time> ms`
  - `Hexagon onCopyBuffer Profile`

## HVX/HMX Guidelines

- For HVX instruction details, read `~/Download/hvx.pdf`.
- Prefer existing project examples before introducing new intrinsic patterns.
- Use `vmem` only when alignment is guaranteed; use `vmemu` for unaligned accesses.
- Keep DMA/HVX/HMX changes grounded in memory traffic and measured op time.
- For HMX paths, verify VTCM allocation sizes, tile counts, and descriptor counts before widening a tile or block.
- Do not assume a micro-optimization is portable across v79/v81; build and test the target architecture.

## DSP DMA-BUF Memory Measurement

Use this when validating Hexagon/DSP memory footprint or comparing CPU vs `forwardtype=10`.

1. Configure the model for DSP:
   - Set the model `config.json` to `forwardtype=10`.
   - Keep the model's intended precision unless the task explicitly asks otherwise.
2. Run enough repeated input to keep the process alive for sampling.
3. Find the process PID:
   - `adb shell ps | grep <process_name>`
4. Sample all three memory views while the process is still running:
   - Process RSS/HWM: `adb shell "awk '/VmRSS:|VmHWM:/{print}' /proc/<PID>/status"`
   - Process DMA-BUF: `adb shell dmabuf_dump <PID>`
   - Full process memory: `adb shell dumpsys meminfo <PID>`
5. Try a system DMA-BUF overview, but record permission/path failures explicitly:
   - `adb shell cat /sys/kernel/debug/dma_buf/bufinfo`
   - Some devices expose `/sys/kernel/dmabuf/buffers`, which may require root.
6. Report memory numbers separately:
   - `VmHWM`
   - `TOTAL RSS`
   - `TOTAL PSS`
   - `dmabuf total` or `PROCESS TOTAL`
   - Approximate process-visible DSP pressure as `VmHWM + dmabuf total` when useful.

## Safety Checklist

- The target model passes correctness checks, such as acceptable `cos_sim`.
- The target demo returns expected output and does not crash.
- `DSPOpType` target timing is reported and compared against a clear baseline.
- `git diff --check` is clean.
- No temporary logs, bucket counters, or per-op tracing remain unless explicitly requested.
- Rebuilt DSP `.so` files are copied/pushed before reporting performance.
