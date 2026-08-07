# SQLite.swift Release checklist

* [ ] Make sure current master branch has a green build
* [ ] Make sure `SQLite.playground` runs without errors
* [ ] Make sure `CHANGELOG.md` is up-to-date
* [ ] Add content to `Documentation/Upgrading.md` if needed
* [ ] Update the version number in `SQLite.swift.podspec`
* [ ] Run `pod lib lint` locally
* [ ] Update the version numbers mentioned in `README.md`, `Documentation/Index.md`
* [ ] Update `MARKETING_VERSION` in `SQLite.xcodeproj/project.pbxproj`
* [ ] Create a tag with the version number (`x.y.z`)
* [ ] Publish to CocoaPods: `pod trunk push`
* [ ] Update the release information on GitHub
