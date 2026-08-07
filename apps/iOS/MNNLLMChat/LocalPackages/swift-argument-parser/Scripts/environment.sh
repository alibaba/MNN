#!/bin/bash
##===----------------------------------------------------------------------===##
##
## This source file is part of the Swift Argument Parser open source project
##
## Copyright (c) 2020 Apple Inc. and the Swift project authors
## Licensed under Apache License v2.0 with Runtime Library Exception
##
## See https://swift.org/LICENSE.txt for license information
##
##===----------------------------------------------------------------------===##

read -p "This will replace your current pasteboard. Continue? [y/n]" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
	swiftversion=$(swift --version)
	unix_version_name=$(uname -a | tr ";" '\n')
	i="${i}Swift version: ${swiftversion}\n"
	i="${i}Unix version: ${unix_version_name}\n"

	# Check if OS is macOS, if so retrieve which version and Xcode version.
	if [[ "$OSTYPE" == "darwin"* ]]; then
		macos=$(defaults read loginwindow SystemVersionStampAsString | cat -)
		xcodebuild_version=$(/usr/bin/xcodebuild -version | grep Xcode)
		xcodebuild_build=$(/usr/bin/xcodebuild -version | grep Build)
		xcodeselectpath=$(xcode-select -p | cat -)
	
		i="${i}\nmacOS version: ${macos}\n"
		i="${i}Xcode-select path: '${xcodeselectpath}\n"
		i="${i}Xcode: ${xcodebuild_version} (${xcodebuild_build})"
	fi

	echo -e "${i}" | pbcopy
	echo "Your pasteboard now contains debug info, paste it on Github"
fi

