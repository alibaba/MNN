# Linux

## Limitations

* Custom functions/aggregations are currently not supported and crash, caused by a bug in Swift.
See [#1071](https://github.com/stephencelis/SQLite.swift/issues/1071).

## Debugging

### Create and launch docker container

```shell
$ docker container create swift:focal
$ docker run --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -i -t swift:focal bash
```

### Compile and run tests in debugger

```shell
$ apt-get update && apt-get install libsqlite3-dev
$ git clone https://github.com/stephencelis/SQLite.swift.git
$ swift test
$ lldb .build/x86_64-unknown-linux-gnu/debug/SQLite.swiftPackageTests.xctest
(lldb) target create ".build/x86_64-unknown-linux-gnu/debug/SQLite.swiftPackageTests.xctest"
(lldb) run
```
