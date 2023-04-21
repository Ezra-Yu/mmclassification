# Copyright (c) OpenMMLab. All rights reserved.
import os

os.system('python -m mim install -e .')

import gradio as gr
from mmpretrain.visualization import UniversalVisualizer
from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain.datasets.categories import IMAGENET_CATEGORIES

visualizer = UniversalVisualizer()
inferencer = ImageClassificationInferencer('vit-base-p16_32xb128-mae_in1k', device='cuda')

def inference_cls(input):
    # test a single image
    result = inferencer(input, return_datasamples=True)[0]
    # show the results
    output = visualizer.visualize_cls(input, result, classes=IMAGENET_CATEGORIES)
    return output

gr.Interface(
    fn=inference_cls,
    inputs=gr.Image(type='numpy'),
    outputs=gr.Image(type='pil'),
    examples=[os.path.join("examples", e) for e in os.listdir("examples")]
).launch()
