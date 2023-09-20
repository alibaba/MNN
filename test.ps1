# Powershell Script must be save as UTF-8 with BOM, otherwise system-wide code page will be used, causing garbled code

# MNN-CPU-GPU
#  |-- include
#  |-- lib
#  |    |-- x64
#  |    |    |-- (Debug/Release x Dynamic/Static x MD/MT)
#  |    |
#  |    |-- x86
#  |         |-- (Debug/Release x Dynamic/Static x MD/MT)
#  |
#  |-- tools (Release + Dynamic + MD)
#  |    |-- x64
#  |    |-- x86
#  |
#  |-- py_whl
#  |-- py_bridge
#       |-- include
#       |-- wrapper
#       |-- test (Release + Dynamic + MD)
#            |-- x64
#            |-- x86
#       |-- lib
#            |-- x64
#            |    |-- (Debug/Release x Dynamic/Static x MD/MT)
#            |
#            |-- x86
#                 |-- (Debug/Release x Dynamic/Static x MD/MT)

Param(
    [Switch]$gpu,
    [Switch]$x86
)

$basedir = $(Split-Path -Parent $MyInvocation.MyCommand.Path)
$outdir = "$basedir/$(If ($gpu) {"MNN-CPU-GPU"} Else {"MNN-CPU"})"
$arch = "$(If ($x86) {"x86"} Else {"x64"})"
Write-Output $arch

$test_avx512 = ((!$x86) -and $env:avx512_server -and $env:avx512_password)
if ($test_avx512) {
    $remote_home = $(Invoke-Expression 'plink -batch -ssh $env:avx512_server -pw $env:avx512_password powershell "echo `$HOME"')
    $remote_dir = "${remote_home}\cise-space\$(Split-Path -Path $(pushd .. ; pwd ; popd) -Leaf)"
}
function sync_remote() {
    Invoke-Expression 'plink -batch -ssh $env:avx512_server -pw $env:avx512_password powershell "Remove-Item -Recurse $remote_dir -ErrorAction Ignore ; mkdir $remote_dir"'
    Invoke-Expression 'pscp -pw $env:avx512_password -r $outdir/tools ${env:avx512_server}:${remote_dir}'
    Invoke-Expression 'pscp -pw $env:avx512_password tools/script/modelTest.py ${env:avx512_server}:${remote_dir}'
}

function run_remote([String]$cmd) {
    $tmpfile = New-TemporaryFile
    Set-Content -Path $tmpfile -Value "powershell `"cd ${remote_dir} ;  $cmd`""
    $output = $(Invoke-Expression 'plink -batch -ssh $env:avx512_server -pw $env:avx512_password -m $tmpfile')
    Remove-Item $tmpfile
    return $output
}

function log($case, $title, $blocked, $failed, $passed, $skipped) {
    Write-Output "TEST_NAME_${case}: $title"
    Write-Output "TEST_CASE_AMOUNT_${case}: {`"blocked`":$blocked,`"failed`":$failed,`"passed`":$passed,`"skipped`":$skipped}"
}

function failed() {
    Write-Output "TEST_NAME_EXCEPTION: Exception"
    Write-Output 'TEST_CASE_AMOUNT_EXCEPTION: {"blocked":0,"failed":1,"passed":0,"skipped":0}'
    exit
}

function build_lib_test() {
    # build_lib_release.ps1 just build release for speed
    Invoke-Expression "./package_scripts/win/build_lib_release.ps1 -path $outdir -cibuild $(If ($gpu) {"-backends 'opencl,vulkan'"}) $(If ($x86) {'-x86'})"
    $WrongNum = [int]$($LastExitCode -ne 0)
    log "WINDOWS_LIB" "Windows主库编译测试" 0 $WrongNum $(1 - $WrongNum) 0
    if ($WrongNum -ne 0) {
        Write-Output "### Windows主库编译测试失败，测试终止"
        failed
    }
}

function build_tool_test() {
    Invoke-Expression "./package_scripts/win/build_tools.ps1 -path $outdir/tools/$arch $(If ($gpu) {"-backends 'opencl,vulkan'"}) -build_all -dynamic_link"
    $WrongNum = $($LastExitCode -ne 0)
    log "WINDOWS_LIB" "Windows工具编译测试" 0 $WrongNum $(1 - $WrongNum) 0
    if ($WrongNum -ne 0) {
        Write-Output "### Windows工具编译测试失败，测试终止"
        failed
    }
}

function build_whl_test() {
    $pyenvs = "py27,py37,py38,py39"
    if ($x86) {
        $pyenvs = "py27-win32,py37-win32,py38-win32,py39-win32"
    }
    Invoke-Expression "./package_scripts/win/build_whl.ps1 -version ci_test -path $outdir/py_whl -pyenvs '$pyenvs' $(If ($x86) {'-x86'})"
    $WrongNum = $($LastExitCode -ne 0)
    log "WINDOWS_LIB" "Windows pymnn wheel编译测试" 0 $WrongNum $(1 - $WrongNum) 0
    if ($WrongNum -ne 0) {
        Write-Output "### Windows pymnn wheel编译测试失败，测试终止"
        failed
    }
}

function build_bridge_test() {
    Invoke-Expression "./package_scripts/win/build_bridge.ps1 -version ci_test -pyc_env py27 -mnn_path $outdir -python_path $HOME/PyBridgeDeps/python -numpy_path $HOME/PyBridgeDeps/numpy -path $outdir/py_bridge -train_api $(If ($x86) {'-x86'})"
    $WrongNum = $($LastExitCode -ne 0)
    log "WINDOWS_LIB" "Windows pymnn bridge编译测试" 0 $WrongNum $(1 - $WrongNum) 0
    if ($WrongNum -ne 0) {
        Write-Output "### Windows pymnn bridge编译测试失败，测试终止"
        failed
    }
}

function unit_test() {
    Invoke-Expression "$outdir/tools/$arch/run_test.out.exe"
    if ($LastExitCode -ne 0) {
        Write-Output "### CPU后端 单元测试失败，测试终止"
        failed
    }
    Invoke-Expression "$outdir/tools/$arch/run_test.out.exe op 0 0 4"
    if ($LastExitCode -ne 0) {
        Write-Output "### CPU后端 多线程测试失败，测试终止"
        failed
    }
    if ($test_avx512) {
        $RemoteExitCode = run_remote "cd tools/x64 ; ./run_test.out.exe > log.txt ; echo `$LastExitCode"
        Write-Output $(run_remote "Get-Content -Path tools/x64/log.txt")
        if ($RemoteExitCode -ne 0) {
            Write-Output "### CPU后端(AVX512) 单元测试失败，测试终止"
            failed
        }
        $RemoteExitCode = run_remote "cd tools/x64 ; ./run_test.out.exe op 0 0 4 > log.txt ; echo `$LastExitCode"
        Write-Output $(run_remote "Get-Content -Path tools/x64/log.txt")
        if ($RemoteExitCode -ne 0) {
            Write-Output "### CPU后端(AVX512) 多线程测试失败，测试终止"
            failed
        }
    }
    #Invoke-Expression "$outdir/tools/$arch/run_test.out.exe op 3"
    #if ($LastExitCode -ne 0) {
    #    echo "### OpenCL后端 单元测试失败，测试终止"
    #    failed
    #}
}

function model_test() {
    Push-Location $outdir/tools/$arch
    python $basedir/tools/script/modelTest.py $HOME/AliNNModel 0 0.002
    if ($LastExitCode -ne 0) {
        Write-Output "### CPU后端 模型测试失败，测试终止"
        Pop-Location
        failed
    }
    python $basedir/tools/script/modelTest.py $HOME/AliNNModel 0 0.002 0 1
    if ($LastExitCode -ne 0) {
        Write-Output "### CPU后端 静态模型测试失败，测试终止"
        Pop-Location
        failed
    }
    if ($test_avx512) {
        $RemoteExitCode = run_remote "cd tools/x64 ; python ../../modelTest.py `$HOME/AliNNModel 0 0.002 > log.txt ; echo `$LastExitCode"
        Write-Output $(run_remote "Get-Content -Path tools/x64/log.txt")
        if ($RemoteExitCode -ne 0) {
            Write-Output "### CPU后端(AVX512) 模型测试失败，测试终止"
            Pop-Location
            failed
        }
        $RemoteExitCode = run_remote "cd tools/x64 ; python ../../modelTest.py `$HOME/AliNNModel 0 0.002 0 1 > log.txt ; echo `$LastExitCode"
        Write-Output $(run_remote "Get-Content -Path tools/x64/log.txt")
        if ($RemoteExitCode -ne 0) {
            Write-Output "### CPU后端(AVX512) 静态模型测试失败，测试终止"
            Pop-Location
            failed
        }
    }
    #python $basedir/tools/script/modelTest.py $HOME/AliNNModel 3 0.01
    #if ($LastExitCode -ne 0) {
    #    echo "### OpenCL后端 模型测试失败，测试终止"
    #    Pop-Location
    #    failed
    #}
    Pop-Location
}

function pymnn_whl_test() {
    $pyarch = $(If ($x86) {"win32"} Else {"amd64"})
    Push-Location pymnn/test
    $local = "$(Get-Location)/aone-site-packages"
    $pythonpath_backup = ${env:PYTHONPATH}
    Foreach ($pyenv in @("27", "37", "38", "39")) {
        Invoke-Expression "conda activate py$pyenv$(If($x86) {'-win32'})"
        Remove-Item -Recurse $local -ErrorAction Ignore
        pip install --target $local $outdir/py_whl/$(Get-ChildItem -Path $outdir/py_whl -Include "*$pyenv*$pyarch*" -Name)
        do {
            # unit_test.py need torch, which isn't support on 32bit Windows and py27
            # https://pytorch.org/docs/stable/notes/windows.html#package-not-found-in-win-32-channel
            if ($x86 -or ($pyenv -eq "27")) {
                break;
            }
            ${env:PYTHONPATH} = $local
            python unit_test.py
            ${env:PYTHONPATH} = $pythonpath_backup
            if ($LastExitCode -ne 0) {
                Write-Output "### PYMNN单元测试失败，测试终止"
                conda deactivate
                Pop-Location
                failed
            }
        } while(0);
        ${env:PYTHONPATH} = "$local"
        python model_test.py $HOME/AliNNModel
        ${env:PYTHONPATH} = $pythonpath_backup
        if ($LastExitCode -ne 0) {
            Write-Output "### PYMNN模型测试失败，测试终止"
            conda deactivate
            Pop-Location
            failed
        }
        conda deactivate
    }
    Pop-Location
}

build_lib_test
# TODO: open other test
# build_tool_test
# build_whl_test
# build_bridge_test

# if ($test_avx512) {
#     sync_remote
# }
# unit_test
# model_test
# pymnn_whl_test