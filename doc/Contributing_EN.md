[中文版本](Contributing_CN.md)

# How to contribute

MNN welcomes issues and pull requests.

# issue

You can submit an issue to report bugs or suggest features. It's recommended to browse existing issues to find a solution before submitting.

# pull request

If you want to contribute codes, you can submit a pull request. Please ensure that the submission meets the following specifications:

- Make sure you have signed the Contributor License Agreement (CLA)
- Make sure you have added license info at the header of each file 
- Read [Code Style Document](CodeStyle_EN.md)
- Execute unit tests

## Header License

You can refer to existing codes or execute [format script](../tools/script/formatLicence.py).

## Execute unit tests

For new backends, new Ops, and new features, you need to add the necessary unit tests under [Test Directory](../test); bugfixs only needs to pass all existing unit tests. Only submit the pull request after passing unit tests.

To compile unit tests, you need to open `MNN_BUILD_TEST` in cmake, so that after make, build products will contain `run_test.out`:

``` bash
cmake .. -DMNN_BUILD_TEST=true
make -j4
./run_test.out # run all tests
./run_test.out unit_test_path_or_prefix # run matching tests
```