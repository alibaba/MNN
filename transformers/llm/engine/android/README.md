# MNN ViT Android Demo

## Build 
Create a signing.gradle at android/app 
template like: 
ext{
    signingConfigs = [
        release: [
            storeFile: file('PATH_TO_jks_file'),
            storePassword: "****",
            keyAlias: "****",
            keyPassword: "****"
        ]
    ]
}

Open with android studio

## launch 
push the model file to android device 
$ adb shell mkdir /data/local/tmp/models
$ adb push <model_folder> /data/local/tmp/models/<model_name>