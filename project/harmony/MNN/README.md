空har模块模板，使用API11，可以fork成为你自己的工程

```
.
├── AppScope
│   ├── app.json5               // 定义了应用程序元数据信息，如应用ID、版本号和权限
│   └── resources               // 存放应用级别的资源文件，比如图像、布局文件等
├── README.md                   // 项目的文档，提供项目概述和使用说明
├── build-profile.json5         // 顶层构建配置文件，定义了整个应用的构建配置
├── entry
│   ├── build                   // 存放entry模块构建产物的目录，通常最终的hap包
│   ├── build-profile.json5     // entry模块的构建配置文件
│   ├── hvigorfile.ts           // entry模块的Hvigor构建任务定义文件
│   ├── obfuscation-rules.txt   // entry模块的代码混淆规则文件
│   ├── oh-package-lock.json5   // 锁定entry模块依赖版本的文件
│   ├── oh-package.json5        // 管理entry模块依赖的配置文件
│   ├── oh_modules              // 存放entry模块依赖模块的目录
│   └── src                     // entry模块的源代码目录
├── hvigor
│   ├── hvigor-config.json5     // Hvigor工具的全局配置文件
│   └── hvigor-wrapper.js       // Hvigor工具的Node.js包装脚本，方便命令行操作
├── hvigorfile.ts               // 顶层目录的Hvigor构建任务定义文件，可能包含全局构建任务
├── hvigorw                     // Unix系统的Hvigor执行脚本，用于快速运行Hvigor任务
├── hvigorw.bat                 // Windows系统的Hvigor执行脚本，用于快速运行Hvigor任务
├── local.properties            // 本地环境的配置文件，可能包括SDK路径和其他本地设置
├── module                      // har包模块，负责打出har包
│   ├── BuildProfile.ets        
│   ├── Index.ets               // 定义模块导出的Ets文件，用于指定模块公开给其他模块使用的API
│   ├── build                   // 存放module模块构建产物的目录，产物为har包
│   ├── build-profile.json5     // module模块的构建配置文件
│   ├── consumer-rules.txt      // 定义如何使用module模块的消费者规则文件
│   ├── hvigorfile.ts           // module模块的Hvigor构建任务定义文件
│   ├── obfuscation-rules.txt   // module模块的代码混淆规则文件
│   ├── oh-package-lock.json5   // 锁定module模块依赖版本的文件
│   ├── oh-package.json5        // 管理module模块依赖的配置文件
│   └── src                     // module模块的源代码目录
├── oh-package-lock.json5       // 锁定顶层依赖版本的文件，确保一致性
├── oh-package.json5            // 顶层管理依赖的配置文件，通常用于使用overrides字段锁定版本
└── oh_modules
    └── @ohos                   // 存放OpenHarmony系统相关的依赖模块的目录，如系统API或框架模块


```

# master分支
纯粹的空模板

# cpp分支
CPP示例+tcpkg依赖管理示例

https://aliyuque.antfin.com/taobao-ohos/uo2wwd/gkshu925mkzzl5x3 《依赖管理》

如果npm报错，需要添加集团仓库

`npm config set @ali:registry https://registry.anpm.alibaba-inc.com`

# api10分支

旧版模板使用api10

