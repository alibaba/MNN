#!/bin/bash
set -ev
if [ -n "$BUILD_SCHEME" ]; then
    if [ -n "$IOS_SIMULATOR" ]; then
        make test BUILD_SCHEME="$BUILD_SCHEME" IOS_SIMULATOR="$IOS_SIMULATOR" IOS_VERSION="$IOS_VERSION"
    else
        make test BUILD_SCHEME="$BUILD_SCHEME"
    fi
elif [ -n "$VALIDATOR_SUBSPEC" ]; then
    bundle install
    if [ "$VALIDATOR_SUBSPEC" == "none" ]; then
      bundle exec pod lib lint --no-subspecs --fail-fast
    else
      bundle exec pod lib lint --subspec="${VALIDATOR_SUBSPEC}" --fail-fast
    fi
elif [ -n "$CARTHAGE_PLATFORM" ]; then
    cd Tests/Carthage && make test CARTHAGE_PLATFORM="$CARTHAGE_PLATFORM"
elif [ -n "$SPM" ]; then
    cd Tests/SPM && swift ${SPM}
elif [ -n "${PACKAGE_MANAGER_COMMAND}" ]; then
    swift ${PACKAGE_MANAGER_COMMAND}
fi
