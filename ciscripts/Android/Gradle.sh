cd project/android/
./gradlew assembleRelease
if [[ -z "${DEPLOY_ENV}" ]]; then
  echo "Android Bintray uploaded due to untrusted CI environment"
else
  ./gradlew bintrayUpload -PbintrayKey=${BINTRAY_DEPLOY_TOKEN}
fi
