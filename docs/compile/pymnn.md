# Pymnn构建
## 本地安装
```bash
cd /path/to/MNN/pymnn/pip_package
python build_deps.py {MNN依赖包组合} #internal,cuda,trt,cuda_tune,opencl,vulkan,render,no_sse,torch这几个字符串的任意组合，例如字符串可为:"cuda,reder,no_sse"
python setup.py install --version {MNN版本} --deps {MNN依赖包组合}
```
## 构建Python Wheel包
- Linux
    ```bash
    # only CPU后端
    ./package_scripts/linux/build_whl.sh -v {MNN版本} -o MNN-CPU/py_whl
    # CPU+OpenCL后端
    ./package_scripts/linux/build_whl.sh -v {MNN版本} -o MNN-CPU-OPENCL/py_whl -b
    ```
- Mac
    ```bash
    # only CPU后端
    ./package_scripts/mac/build_whl.sh -v {MNN版本} -o MNN-CPU/py_whl -p py27,py37,py38,py39
    # CPU+OpenCL后端
    ./package_scripts/mac/build_whl.sh -v {MNN版本} -o MNN-CPU/py_whl -p py27,py37,py38,py39 -b
    ```
- Windows
    ```bash
    # CPU，64位编译
    powershell .\package_scripts\win\build_whl.ps1 -version {MNN版本} -path MNN-CPU/py_whl/x64 -pyenvs "py27,py37,py38,py39"
    # CPU，32位编译
    powershell .\package_scripts\win\build_whl.ps1 -version {MNN版本} -x86 -path MNN-CPU/py_whl/x86 -pyenvs "py27-win32,py37-win32,py38-win32,py39-win32"

    # CPU+OpenCL，64位编译
    .\package_scripts\win\build_whl.ps1 -version {MNN版本} -backends opencl -path MNN-CPU-OPENCL/py_whl/x64 -pyenvs "py27,py37,py38,py39"
    # CPU+OpenCL，32位编译
    .\package_scripts\win\build_whl.ps1 -version {MNN版本} -backends opencl -x86 -path MNN-CPU-OPENCL/py_whl/x86 -pyenvs "py27-win32,py37-win32,py38-win32,py39-win32"

    # CPU+Vulkan，64位编译
    .\package_scripts\win\build_whl.ps1 -version {MNN版本} -backends vulkan -path MNN-CPU-OPENCL/py_whl/x64 -pyenvs "py27,py37,py38,py39"
    # CPU+Vulkan，32位编译
    .\package_scripts\win\build_whl.ps1 -version {MNN版本} -backends vulkan -x86 -path MNN-CPU-OPENCL/py_whl/x86 -pyenvs "py27-win32,py37-win32,py38-win32,py39-win32"

    # CPU+OpenCL+Vulkan，64位编译
    .\package_scripts\win\build_whl.ps1 -version {MNN版本} -backends "opencl,vulkan" -path MNN-CPU-OPENCL/py_whl/x64 -pyenvs "py27,py37,py38,py39"
    # CPU+OpenCL+Vulkan，32位编译
    .\package_scripts\win\build_whl.ps1 -version {MNN版本} -backends "opencl,vulkan" -x86 -path MNN-CPU-OPENCL/py_whl/x86 -pyenvs "py27-win32,py37-win32,py38-win32,py39-win32"
    ```
