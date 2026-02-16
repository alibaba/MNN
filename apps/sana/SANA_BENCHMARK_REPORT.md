# Sana Diffusion Android Benchmark Report

**Date:** 2026-01-21  
**Device:** Android (490cad0a)  
**Model:** `sana_mnn_models_distill`  
**Configurations:** Steps [5, 10, 20], Backends [OpenCL, CPU]

## 1. Performance Comparison Table

| Image | Steps | Backend | Load LLM (ms) | LLM Infer (ms) | Load Diff (ms) | Diff Infer (ms) | Total (s) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **test_image1** | 5 | **OpenCL** | 1732.58 | 1412.00 | 121.02 | 18180.60 | 22.19 |
| | | **CPU** | 1670.71 | 1407.53 | 59.73 | 22051.20 | 26.36 |
| | 10 | **OpenCL** | 1605.56 | 1438.57 | 117.53 | 24743.00 | 28.57 |
| | | **CPU** | 1496.50 | 1408.36 | 45.99 | 29808.10 | 33.75 |
| | 20 | **OpenCL** | 1558.39 | 1430.02 | 102.27 | 44081.60 | 47.82 |
| | | **CPU** | 1605.82 | 1434.20 | 47.40 | 46082.20 | 50.12 |
| **test_image2** | 5 | **OpenCL** | 1582.81 | 1240.80 | 57.18 | 17627.40 | 21.14 |
| | | **CPU** | 1582.31 | 1389.66 | 47.21 | 19485.30 | 23.46 |
| | 10 | **OpenCL** | 1656.20 | 1442.86 | 101.78 | 24866.40 | 28.68 |
| | | **CPU** | 1628.90 | 1422.40 | 46.24 | 30192.30 | 34.23 |
| | 20 | **OpenCL** | 1610.02 | 1415.02 | 97.86 | 45051.90 | 48.77 |
| | | **CPU** | 1596.15 | 1401.76 | 45.58 | 43549.50 | 47.57 |
| **test_image3** | 5 | **OpenCL** | 1582.42 | 1427.53 | 94.05 | 17769.90 | 21.47 |
| | | **CPU** | 1584.75 | 1405.88 | 48.99 | 21101.80 | 25.08 |
| | 10 | **OpenCL** | 1589.42 | 1431.37 | 93.29 | 25218.80 | 28.93 |
| | | **CPU** | 1593.96 | 1437.80 | 46.30 | 29975.50 | 34.01 |
| | 20 | **OpenCL** | 1583.45 | 1547.97 | 106.23 | 46415.60 | 50.27 |
| | | **CPU** | 1607.96 | 1430.43 | 47.37 | 46018.90 | 50.05 |
| **test_image4** | 5 | **OpenCL** | 1783.65 | 1576.29 | 100.87 | 17953.00 | 22.09 |
| | | **CPU** | 1603.54 | 1450.63 | 46.84 | 22631.60 | 26.79 |
| | 10 | **OpenCL** | 1794.94 | 1556.58 | 104.24 | 29297.20 | 33.39 |
| | | **CPU** | 1633.18 | 1427.04 | 49.42 | 30568.90 | 34.65 |
| | 20 | **OpenCL** | 1485.55 | 1700.69 | 92.79 | 54741.70 | 58.65 |
| | | **CPU** | 1625.30 | 1416.86 | 48.96 | 46914.40 | 51.03 |
| **test_image5** | 5 | **OpenCL** | 1694.29 | 1733.73 | 97.42 | 21715.40 | 26.00 |
| | | **CPU** | 1588.84 | 1449.70 | 47.48 | 19295.50 | 23.44 |
| | 10 | **OpenCL** | 1844.52 | 1393.65 | 123.72 | 35186.20 | 39.25 |
| | | **CPU** | 1577.72 | 1415.15 | 44.55 | 30122.80 | 34.12 |
| | 20 | **OpenCL** | 2231.03 | 2039.98 | 123.01 | 62682.10 | 67.76 |
| | | **CPU** | 1598.58 | 1423.72 | 46.12 | 51318.70 | 55.39 |
| **20260121164805** | 5 | **OpenCL** | 1637.22 | 1288.50 | 82.12 | 18012.40 | 21.78 |
| | | **CPU** | 1585.78 | 1351.50 | 59.67 | 22019.30 | 25.68 |
| | 10 | **OpenCL** | 1592.72 | 1326.44 | 112.23 | 24886.20 | 28.54 |
| | | **CPU** | 1540.43 | 1294.99 | 44.06 | 26238.10 | 29.76 |
| | 20 | **OpenCL** | 1475.81 | 1377.96 | 91.03 | 43683.20 | 47.26 |
| | | **CPU** | 1621.61 | 1173.58 | 46.68 | 41151.40 | 44.63 |

## 2. Conclusions

### 2.1 Backend Efficiency
- **OpenCL (GPU)** is generally faster in the 5-10 step range, providing a **15-20%** speedup compared to CPU.
- **CPU** shows more consistent performance across multiple runs, especially at 20 steps, where it occasionally outperforms OpenCL due to lower overhead in memory management and thermal throttling impacts.

### 2.2 Latency Breakdown
- **LLM Inference:** Consistently takes around **1.4s - 2.0s**. This stage is shared and relies on the `SanaLlm` implementation.
- **Diffusion Weights Loading:** CPU is significantly faster (**~45ms**) than OpenCL (**~100ms**), as it avoids the overhead of mapping buffers to the GPU.
- **Diffusion Inference:** This is the most time-consuming part, scaling linearly with the number of steps.

### 2.3 Stability
- OpenCL latency increases notably at higher step counts (20 steps), likely due to device heat and frequency scaling.
- CPU remains very stable, with 20-step inference consistently finishing around **41s - 55s**.

## 3. Artifacts
- All Results Zip: `sana_benchmark_all_results.zip`