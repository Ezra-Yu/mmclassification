# 分类模型部署

[MMDeploy](https://github.com/open-mmlab/mmdeploy) 是 OpenMMLab 的部署仓库，负责包括 MMPreTrain、MMDetection 等在内的各算法库的部署工作。
你可以从[这里](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpretrain.html#mmpretrain-deployment)获取 MMDeploy 对 MMPreTrain 部署支持的最新文档。

本文结构如下：

- [安装](#安装)
- [部署onnxruntime](#部署onnxruntime)
  - [模型转换](#模型转换])
  - [后端模型推理](#后端模型推理)
  - [SDK 模型推理](#sdk-模型推理)
- [其他后端](#模型规范)
- [模型支持列表](#模型支持列表)

## 安装

- 安装 MMPreTrain, 参考[教程](https://mmpretrain.readthedocs.io/zh_CN/latest/get_started.html)。

- 安装 MMDeploy, 参考[教程](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy)。

## 部署 onnxruntime

### 1.模型转换

以 [resnet18](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnet18_8xb32_in1k.py) 在 ImageNet-1k 数据集上的预训练模型为例：

新建一个 'classification_onnxruntime_dynamic.py' 的文件，将下列内容复制进去。

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

通过以下步骤导出 onnxruntime 模型：

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

# 1. 将模型从 torch 转化为 onnx 格式
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg, model_checkpoint, device)

# 2. MMDeploy SDK 提取模型推理的前处理
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
```

**导出模型规范**

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。

它存放在 `--work-dir` 指定的路路径下, 比如上例中的 `mmdeploy_models/cls/onnx`，其结构如下：

```
mmdeploy_models/mmpretrain/onnx
            ├── deploy.json
            ├── detail.json
            ├── end2end.onnx    # 推理引擎文件。可用 ONNX Runtime 推理
            └── pipeline.json
```

其中：

- **end2end.onnx**: 推理引擎文件。可用 ONNX Runtime 推理
- ***xxx*.json**:  mmdeploy SDK 推理所需的 meta 信息

整个文件夹被定义为**mmdeploy SDK model**，既包括推理引擎，也包括推理 meta 信息。

### 2.后端模型推理

以上述模型转换后的 `end2end.onnx` 为例，你可以使用如下代码进行推理：

```python
from mmdeploy.apis import inference_model

model_cfg = 'configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'classification_onnxruntime_dynamic.py'
backend_files = ['mmdeploy_models/cls/onnx/end2end.onnx']
img = './demo/demo.JPEG'
device = 'cpu'

result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
```

### 3. SDK 模型推理

你也可以参考如下代码，对 SDK model 进行推理：

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

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/main/demo)学习其他语言接口的使用方法。

## 转换成其他推理后端

目前已经支持了以下后端：

- onnxruntime
- tensorrt
- pplnn
- ncnn
- openvino
- coreml

使用其他后端推理时，需要安装后端的 MMDeploy，详细请参考 MMDeploy 的[安装文档](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy)。

转换的关键之一是使用正确的配置文件, MMDeploy 中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpretrain)，它们的命名格式为：

```
classification_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等
- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32
- **{static | dynamic}:** 动态、静态 shape
- **{shape}:** 模型输入的 shape 或者 shape 范围

比如，你也可以把模型转为 tensorrt-fp16 模型，就要使用 `classification_tensorrt-fp16_dynamic-224x224-224x224.py` 配置文件。

```{note}
当转 tensorrt 模型时, --device 需要被设置为 "cuda"
```

## 模型支持列表

请参考[这里](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpretrain.html#supported-models)
