# Model Quantization
## Advantages of quantization
Quantization can accelerate forward speed of the model by converting floating point computations in the original model into int8 computations. At the same time, it compresses the original model by approximately 4X by quantize the float32 weights into int8 weights.

## Compile
### Compile macro
In order to build the quantization tool, set `MNN_BUILD_QUANTOOLS=true` when compiling, like this:

```bash
cd MNN
mkdir build
cd build
cmake -DMNN_BUILD_QUANTOOLS=ON ..
make -j4
```

## Usage
### Command
```bash
./quantized.out origin.mnn quan.mnn preprocessConfig.json
```

- The first argument is the path of floating point model to be quantized

- The second argument indicates the saving path of quantized model
- The third argument is the path of config json file

### Json config file

```bash
{
    "format":"RGB",
    "mean":[
        127.5,
        127.5,
        127.5
    ],
    "normal":[
        0.00784314,
        0.00784314,
        0.00784314
    ],
    "width":224,
    "height":224,
    "path":"path/to/images/",
    "used_image_num":500,
    "feature_quantize_method":"KL",
    "weight_quantize_method":"MAX_ABS"
}
```

#### format
The format of input images is RGBA, then converted to target format specified by `format`.

>  Options: "RGB", "BGR", "RGBA", "GRAY"

#### mean, normal
The same as ImageProcess config

$dst = (src - mean) * normal$

#### width, height
Input width and height of the floating point model

#### path
Path to images that are used for calibrating feature quantization scale factors.

#### used_image_num
Specify the number of images used for calibration.

>  Default: use all the images under `path`.

>  *Note: please confirm that the data after the images are transformed by the above processes are the exact data that fed into the model input.*

#### feature_quantize_method
Specify method used to compute feature quantization scale factor.

Options: 

- "KL": use KL divergence method, generally need 100 ~ 1000 images.

- "ADMM": use ADMM (Alternating Direction Method of Multipliers) method to iteratively search for optimal feature quantization scale factors, generally need one batch images.

>  Default: "KL"

#### weight_quantize_method
Specify weight quantization method

Options:

- "MAX_ABS": use the max absolute value of weights to do symmetrical quantization.

- "ADMM": use ADMM method to iteratively find optimal quantization of weights.

> Default: "MAX_ABS"

Users can explore the above feature and weight quantization methods, and choose a better solution.

## Usage of quantized model
The same as floating point model. The inputs and outputs of quantized model are also floating point.
