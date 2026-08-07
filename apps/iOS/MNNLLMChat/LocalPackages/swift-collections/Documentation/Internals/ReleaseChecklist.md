# Swift Collections Release Checklist

1. Create a milestone for the new version (if one doesn't exist yet).
2. Collect all issues & PRs that are going to be included in the new tag under the new milestone.
3. If the new release moves code between source files or adds new API that has the potential to cause mutual dependencies between source files, then run the [shuffle-sources.sh](./Utils/shuffle-sources.sh) script for at least a few hundred iterations on the affected module to help catch [nondeterministic build issues with the compiler's MergeModules phase](https://github.com/apple/swift-collections/issues/7). (Note: it's best to do this on a fresh clone. We can stop doing this when MergeModules is no longer used to build debug configurations in SPM.)
4. Run the [full tests script](./Utils/run-full-tests.sh) on the commit that you intend to tag, with all supported (major) toolchain releases, and on as many supported platforms as are practical. (At minimum, run the script with the latest stable toolchain releases on macOS and one Linux distribution.)
  The script exercises a tiny subset of the environments this package can be built and run on.
  
   The full matrix includes the following axes:

    - Toolchains
      - All major Swift releases supported by the package, plus a recent toolchain snapshot from swift.org.
        E.g., for the Swift Collection 1.0 release, this included:
        - Swift 5.3 (from swift.org and Xcode 12)
        - Swift 5.4 (from swift.org and Xcode 12.5)
        - Prerelease builds of Swift 5.5 (using the latest swift.org snapshot and the latest Xcode 13 beta)
        - The latest `main` development snapshot from swift.org.
    - Platforms & architectures
      - macOS (using SPM, xcodebuild, cmake)
          - Intel, Apple Silicon
          - Deploying on any macOS release starting from macOS 10.10
      - Mac Catalyst (using xcodebuild)
          - Intel, Apple Silicon
          - Deploying on any macOS release starting from macOS 10.15
      - iOS (using xcodebuild)
          - Device (arm64, armv7), simulator (x86_64, i386, arm64)
          - Deploying on any iOS release starting from iOS 8
      - watchOS (using xcodebuild)
          - Device (arm64_32, arm7k), simulator (x86_64, i386, arm64)
          - Deploying on any watchOS release starting from watchOS 2
      - tvOS (using xcodebuild)
          - Device (arm64), simulator (x86_64, arm64)
          - Deploying on any tvOS 9 release starting from tvOS 10
      - Linux (using SPM, cmake)
          - All supported distributions
    - Build systems
      - Swift Package Manager
      - xcodebuild
      - cmake & ninja (note: this support isn't source stable)
    - Configurations
      - Debug
      - Release
    - Build settings
      - `COLLECTIONS_INTERNAL_CHECKS`
      - `COLLECTIONS_DETERMINISTIC_HASHING`
      - `BUILD_LIBRARY_FOR_DISTRIBUTION=YES` with xcodebuild (this is unsupported, but why break it)
      - `-warnings-as-errors` 
      - Any combination of the above
    - Components
      - The swift-collections package
      - The private `benchmark` executable under `Benchmarks/`

5. Submit PRs to fix any issues that pop up, then try again until no issues are found.
6. Update the README if necessary.
7. Draft a new release on GitHub, following the template established by [previous releases](https://github.com/apple/swift-collections/releases/tag/0.0.7).
8. As a final chance to catch issues, generate a diff between the last tagged release and the release candidate commit. Review it in case we landed something that isn't appropriate to include; watch out for potential issues such as source compatibility problems, or new public API without adequate documentation.
9. Double check that the new tag will have the right version number and it will be on the correct branch.
10. Hit publish.
