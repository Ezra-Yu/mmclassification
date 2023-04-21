# Copyright (c) OpenMMLab. All rights reserved.
import os

os.system('python -m mim install "mmcv>=2.0.0"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system('python -m mim install "mmcls>=1.0.0"')
os.system('python -m mim install -e .')

from argparse import ArgumentParser

import gradio as gr

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def inference(input):
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('--config', default='configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py', help='Config file') #noqa
    parser.add_argument('--checkpoint', default='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth', help='Checkpoint file') #noqa
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, input)
    # show the results
    output = show_result_pyplot(
        model,
        input,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False,
        out_file=args.out_file) 
    return output

gr.Interface(
    fn=inference,
    inputs=gr.Image(type='numpy'),
    outputs=gr.Image(type='pil'),
    examples=['demo/demo.png']
).launch()
