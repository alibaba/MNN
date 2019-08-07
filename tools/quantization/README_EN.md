# Model Quantization
## Advantages of quantization
Quantization can accelerate forward speed of the model by converting floating point computations in the original model into int8 computations. At the same time, it compresses the original model by approximately 4X by quantize the float32 weights into int8 weights.

## Compile
### Compile macro
In order to build the quantization tool, set `MNN_BUILD_QUANTOOLS=true` when compiling.

### Compile outputs
Quantization tool: `quantized.out`<br>
Comparison tool(between floating point model and int8 quantized model): `testQuanModel.out`

## Usage
### Command
```bash
./quantized.out origin.mnn quan.mnn pretreatConfig.json
```

The first argument is the path of floating point model to be quantized.<br>
The second argument indicates the saving path of quantized model.<br>
The third argument is the path of config json file.

### Json config file
#### format
Images are read as RGBA format, then converted to target format specified by `format`.<br>
Options: "RGB", "BGR", "RGBA", "GRAY"

#### mean normal
The same as ImageProcess config<br>
dst = (src - mean) * normal

#### width, height
Input width and height of the floating point model

#### path
Path to images that are used for calibrating feature quantization scale factors.

#### used_image_num
Specify the number of images used for calibration.<br>
Default: use all the images under `path`.

*Note: please confirm that the data after the images are transformed by the above processes are the exact data that fed into the model input.*

#### feature_quantize_method
Specify method used to compute feature quantization scale factor.<br>
Options:
1. "KL": use KL divergence method, generally need 100 ~ 1000 images.
2. "ADMM": use ADMM (Alternating Direction Method of Multipliers) method to iteratively search for optimal feature quantization scale factors, generally need one batch images.

default: "KL"

#### weight_quantize_method
Specify weight quantization method<br>
Options:
1. "MAX_ABS": use the max absolute value of weights to do symmetrical quantization.
2. "ADMM": use ADMM method to iteratively find optimal quantization of weights.

default: "MAX_ABS"

Users can explore the above feature and weight quantization methods, and choose a better solution.

## Usage of quantized model
The same as floating point model. The inputs and outputs of quantized model are also floating point.
