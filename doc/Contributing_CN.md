[English Version](Contributing_EN.md)

# 如何贡献

MNN欢迎提交issue和pull request。

# issue

如果你有bug反馈、feature建议，可以提交issue反馈。建议在提交之前，先浏览已有issue，寻找解决方案。

# pull request

如果你想贡献代码，可以提交pull request。请确保提交满足下列规范：

- 确保你已签署Contributor License Agreement(CLA)
- 代码文件头License
- 浏览[代码风格文档](CodeStyle_CN.md)，统一代码风格
- 执行单元测试

## 代码文件头License

可以参考已有代码，或执行[规范化脚本](../tools/script/formatLicence.py)。

## 执行单元测试

对于新增Backend、新增Op、新增Feature，需要在[测试目录](../test)下增加必要的单元测试；bugfix则只需回归已有单元测试即可。确保单元测试通过后再提交pull request。

要编译单元测试，需要在cmake时开启`MNN_BUILD_TEST`，这样，make之后，产物中就会包含`run_test.out`：

``` bash
cmake .. -DMNN_BUILD_TEST=true
make -j4
./run_test.out                            # run all tests
./run_test.out unit_test_path_or_prefix   # run matching tests
```
