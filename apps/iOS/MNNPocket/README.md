# MNN Pocket iOS demo

MNN Pocket is a separate iOS app from `../MNNLLMChat`. Open
`MNNPocket.xcodeproj` and select the `MNNPocket` scheme. Its bundle identifier,
display name, entry point, model bundle, and local Swift packages are kept in
this directory, so changes to this demo do not replace the MNN Chat app.

The app discovers MNN-format models in `MNNPocket/LocalModel/` and opens a
single-model chat after the user selects one. The model weights, `MNN.framework`,
and `libsherpa-mnn.a` are local build assets and are not committed to Git.
Provision the framework and speech library with the same MNN revision used to
export the models. For framework build steps, see `../MNNLLMChat/README-ZH.md`.

All shared chat code comes from the local Swift package `../MNNChatCore`
(used by both iOS demo apps); this target only carries the entry point,
signing, assets and bundled models. After updating `MNN.framework` or
`libsherpa-mnn.a`, regenerate the package's binary dependencies:

```sh
sh apps/iOS/Scripts/build-xcframeworks.sh
```

Build for a device from the repository root:

```sh
env -i PATH=/usr/bin:/bin xcodebuild \
  -project apps/iOS/MNNPocket/MNNPocket.xcodeproj \
  -scheme MNNPocket -configuration Debug \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
```

For a real-device launch, select your own signing team in Xcode, install the
`MNNPocket` scheme, and verify that a bundled model loads and answers a prompt.
