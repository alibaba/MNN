# Benchmark data from the swift.org announcement

This directory contains the benchmark configuration that we used to collect benchmarking data and render the charts in the [Swift Collections announcement on swift.org][announcement].

* [`generate-results.sh`](./generate-results.sh): A script that you can use to run the benchmark on your own machine, and to reproduce the charts.
* [`Library.json`](./Library.json): The benchmark library definition collecting the chart definitions used in the blog post.
* [`Theme.json`](./Theme.json): The chart theme to set up colors, line widths, font sizes etc. (Slightly adapted.)
* [`Results.md`](./Results.md): The generated results summary.
* [`Results/`](./Results/): The generated subdirectory containing the PNG files for the charts.
* [`results.json`](./results.json): The result data we collected.

Note: If you'd like to try reproducing our results (it's easy!), note that the script is configured to collect 20 rounds of data on up to 16 million items, and this will likely take a very long time. Feel free to reduce the maximum size, or just plan to allow the measurement run overnight.

For more information on how to use the Swift Collections benchmarking tool, please see [Swift Collections Benchmark][swift-collections-benchmark].

[announcement]: https://swift.org/blog/swift-collections
[swift-collections-benchmark]: https://github.com/apple/swift-collections-benchmark
