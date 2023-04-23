# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable
os.system('python -m mim install -e .')

import gradio as gr
from mmpretrain.apis import ImageClassificationInferencer
from functools import partial
from mmpretrain.apis import list_models

class InferencerCache:
    max_size = 2
    _cache = []

    @classmethod
    def get_instance(cls, instance_name, callback: Callable):
        if len(cls._cache) > 0:
            for i, cache in enumerate(cls._cache):
                if cache[0] == instance_name:
                    # Re-insert to the head of list.
                    cls._cache.insert(0, cls._cache.pop(i))
                    return cache[1]

        if len(cls._cache) == cls.max_size:
            cls._cache = cls._cache[:cls.max_size - 1]
        instance = callback()
        cls._cache.insert(0, (instance_name, instance))
        return instance



class ImageClassificationTab:

    def __init__(self) -> None:
        self.short_list = [
            'vit-base-p16_32xb128-mae_in1k',
            'vit-base-p16_mae-1600e-pre_8xb128-coslr-100e_in1k',
            'vit-huge-p14_mae-1600e-pre_32xb8-coslr-50e_in1k-448px',
            'vit-huge-p14_mae-1600e-pre_8xb128-coslr-50e_in1k',
            'vit-large-p16_mae-1600e-pre_8xb128-coslr-50e_in1k']
        # self.long_list = list_models("vit*mae*_in1k")
        self.tab = self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_classification_models',
                    elem_classes='select_model',
                    choices=self.short_list,
                    value='vit-base-p16_32xb128-mae_in1k',
                )
                in_image = gr.Image(
                    value=None,
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor'
                )
                gr.Examples(
                    examples=[os.path.join("examples", e) for e in os.listdir("examples")],
                    inputs = in_image,
                    outputs=in_image
                )

            with gr.Column():
                out_cls = gr.Label(
                    label='Result',
                    num_top_classes=5,
                    elem_classes='cls_result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, in_image],
                    outputs=out_cls,
                )

    def inference(self, model, image):
        inferencer_name = self.__class__.__name__ + model
        inferencer = InferencerCache.get_instance(
            inferencer_name, partial(ImageClassificationInferencer, model))
        result = inferencer(image)[0]['pred_scores'].tolist()

        if inferencer.classes is not None:
            classes = inferencer.classes
        else:
            classes = list(range(len(result)))

        return dict(zip(classes, result))

if __name__ == '__main__':
    title = 'MAE Inference Demo in MMPretrain '
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(f'# {title}')
        with gr.Tabs():
            with gr.TabItem('Image Classification'):
                ImageClassificationTab()
            # with gr.TabItem('Masked Image Recovery'):
            #     pass

    demo.launch(server_name='0.0.0.0')

                                  