## 开发规范
* 需要判断是否用到NEON时，使用`#ifdef MNN_USE_NEON`。不要直接判断是否为arm，有些arm是不支持neon的。

## 交叉编译
* 在项目根目录中调用`./project/cross-compile/build.sh`。
