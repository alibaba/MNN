#$erroractionpreference = "stop"

pushd (get-item $MyInvocation.MyCommand.Definition).Directory.FullName

if (($args[0] -eq "-lazy") -and ( Test-Path "current" -PathType Container )) {
  popd
  echo "*** done ***"
  exit
}

# check is flatbuffer installed or not
Set-Variable -Name "FLATC" -Value "..\3rd_party\flatbuffers\tmp\flatc.exe"
if (-Not (Test-Path $FLATC -PathType Leaf)) {
  echo "*** building flatc ***"

  # make tmp dir
  pushd ..\3rd_party\flatbuffers
  if (-Not (Test-Path "tmp" -PathType Container)) {
    mkdir tmp
  }
  (cd tmp) -and (rm -r -force *)

  # build
  cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --target flatc

  # dir recover
  popd
}

# determine directory to use\
Set-Variable -Name "DIR" -Value "default"
if (Test-Path "private" -PathType Container) {
  Set-Variable -Name "DIR" -Value "private"
}

# clean up
echo "*** cleaning up ***"
if (-Not (Test-Path "current" -PathType Container)) {
  mkdir current
}
rm -force current\*.h

# flatc all fbs
pushd current
echo "*** generating fbs under $DIR ***"
Get-ChildItem ..\$DIR\*.fbs | %{Invoke-Expression "..\$FLATC -c -b --gen-object-api --reflect-names  $_"}
popd

# finish
popd
echo "*** done ***"
