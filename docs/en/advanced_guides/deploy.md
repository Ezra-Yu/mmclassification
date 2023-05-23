# Deployment of Classification Models

[MMDeploy](https://github.com/open-mmlab/mmdeploy) is OpenMMLab's deployment repository, responsible for the deployment of various algorithm libraries including MMPreTrain, MMDetection and more. You can get the latest documentation on MMPreTrain deployment support from [here](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpretrain.html#mmpretrain-deployment).

The structure of this article is as follows:

- [Installation](#installation)
- [Deploy onnxruntime](#deploy-onnxruntime)
  - [Model conversion](#model-conversion)
  - [Backend model inference](#backend-model-inference)
  - [SDK model inference](#sdk-model-inference)
- [Other backends](#model-specification)
- [Model support list](#model-support-list)

## Installation

- Install MMPreTrain, refer to the [tutorial](https://mmpretrain.readthedocs.io/zh_CN/latest/get_started.html).

- Install MMDeploy, refer to the [tutorial](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy).

## Deploy onnxruntime

### 1.Model conversion

Taking [resnet18](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnet18_8xb32_in1k.py) pre-trained model on ImageNet-1k dataset as an example:

Create a new file 'classification_onnxruntime_dynamic.py' and copy the following content into it.

```{python}
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None,
    optimize=True,
    dynamic_axes=dict(
        input=dict({
            0: 'batch',
            2: 'height',
            3: 'width'
        }),
        output=dict({0: 'batch'})))
codebase_config = dict(type='mmpretrain', task='Classification')
backend_config = dict(type='onnxruntime')
```

Export onnxruntime model with the following steps:

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'demo/demo.JPEG'
work_dir = 'mmdeploy_models/cls/onnx'
save_file = 'end2end.onnx'
deploy_cfg = 'classification_onnxruntime_dynamic.py'
model_cfg = 'configs/resnet/resnet18_8xb32_in1k.py'
model_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
device = 'cpu'

# 1. Convert the model from torch to onnx format
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg, model_checkpoint, device)

# 2. Extract preprocessing of model inference from MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
```

**Exporting model specification**

It is necessary to understand the structure of the conversion result before using the converted model for inference.

It is stored in the path specified by `--work-dir`, such as `mmdeploy_models/cls/onnx` in the above example, with the following structure:

```
mmdeploy_models/mmpretrain/onnx
            ├── deploy.json
            ├── detail.json
            ├── end2end.onnx    # Inference engine file. Can be inferred by ONNX Runtime
            └── pipeline.json
```

Where:

- **end2end.onnx**: Inference engine file. Can be inferred by ONNX Runtime
- ***xxx*.json**: Meta information required for mmdeploy SDK inference

The entire folder is defined as **mmdeploy SDK model** which includes both the inference engine and meta information for inference.

### 2.Backend model inference

Using the `end2end.onnx` converted from the above model as an example, you can use the following code for inference:

```python
from mmdeploy.apis import inference_model

model_cfg = 'configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'classification_onnxruntime_dynamic.py'
backend_files = ['mmdeploy_models/cls/onnx/end2end.onnx']
img = './demo/demo.JPEG'
device = 'cpu'

result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
```

Translate the following content into English:

### 3. SDK Model Inference

You can also refer to the following code to perform inference on the SDK model:

```python
from mmdeploy_runtime import Classifier
import cv2

# read the image
img = cv2.imread('demo/cat-dog.png')
# create a classifier
classifier = Classifier(model_path='mmdeploy_models/mmcls/onnx', device_name='cpu', device_id=0)
# perform inference
result = classifier(img)
```

In addition to the Python API, the mmdeploy SDK also provides interfaces in multiple languages such as C, C++, C#, Java, etc.
You can refer to [examples](https://github.com/open-mmlab/mmdeploy/tree/main/demo) for guidance on using these interfaces in other languages.

## Convert to Other Inference Backends

The following backends are currently supported:

- onnxruntime
- tensorrt
- pplnn
- ncnn
- openvino
- coreml

When using other backend inference, you need to install the MMDeploy for that backend. For more information, please refer to MMDeploy's [installation documentation](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy).

One of the keys to conversion is using the correct configuration file. MMDeploy has built-in deployment [configuration files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpretrain) for each backend, with the naming format:

```
classification_{backend}-{precision}_{static | dynamic}_{shape}.py
```

Where:

- **{backend}:** Inference backend name. For example, onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml, etc.
- **{precision}:** Inference precision. For example, fp16, int8. If not specified, it defaults to fp32
- **{static | dynamic}:** Static or dynamic shape
- **{shape}:** The input shape of the model or the shape range

For example, you can also convert the model to a tensorrt-fp16 model using the `classification_tensorrt-fp16_dynamic-224x224-224x224.py` configuration file.

```{note}
When converting to a TensorRT model, the --device flag should be set to "cuda"
```

## Supported Model List

Please refer to [this link](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpretrain.html#supported-models)
