./schema/generate.sh
xcodebuild -configuration Release -project project/ios/MNN.xcodeproj
find . -name ".DS_Store" -delete
cd project/ios/build/Release-iphoneos/
zip -r MNN.iOS.framework.zip ./
if [[ -z "${DEPLOY_ENV}" ]]; then
  echo "iOS Bintray uploaded due to untrusted CI environment"
else
  curl -T MNN.iOS.framework.zip -umnn:${BINTRAY_DEPLOY_TOKEN} https://api.bintray.com/content/mnnteam/Pods/Nightly/0.0.0/MNN-iOS-Nightly.zip
fi
