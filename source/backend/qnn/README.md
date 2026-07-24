# QNN intermediate tensor dump

The QNN backend can expose every native activation as an application-readable
QNN graph output. This follows the same mechanism as ExecuTorch's QNN
intermediate debugger: native tensors become `QNN_TENSOR_TYPE_APP_READ` before
the graph is finalized, receive host buffers, and are returned by
`graphExecute`. The dump contains the complete `graphExecute` output set, including
the model outputs and promoted intermediate tensors.

This mode is intended only for accuracy debugging. It increases graph outputs,
memory use, and execution time.

## Online QNN graph

Set the QNN flag through the normal MNN backend configuration:

```cpp
MNN::BackendConfig backendConfig;
backendConfig.flags = MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS;

MNN::ScheduleConfig schedule;
schedule.type = MNN_FORWARD_NN;
schedule.backendConfig = &backendConfig;
```

By default, files are written to `./qnn_intermediate_outputs`. Set
`MNN_QNN_DUMP_DIR` before creating any QNN runtime to choose another directory.

## Serialized QNN graph

A finalized QNN context cannot expose tensors that were native when the context
was built. Generate a separate debug artifact:

```bash
MNN2QNNModel /path/to/qnn/sdk 57 75 model.mnn output \
  --dump_intermediate_outputs
```

The flag may appear anywhere among the optional dynamic-shape arguments.

The generated MNN plugin model remembers that it is a debug artifact and writes
files to `qnn_intermediate_outputs` beside the model by default. Release
artifacts generated without the option are unchanged.

## Output format

Each execution creates one `manifest_NNNNNN.tsv` and one raw file per readable
tensor. The manifest records:

- QNN tensor name
- raw file name
- QNN data-type value
- dimensions in QNN layout
- quantization encoding, scale, and offset

MNN graph tensors retain names such as `t42`, allowing tools to map them back
to the model tensor table. Backend-created stages retain operation-derived
names. Raw values remain in QNN layout and data type; comparison tools should
apply the manifest's quantization and layout metadata before comparing them
with CPU tensors.
