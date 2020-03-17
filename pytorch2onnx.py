#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

import pickle
#from resnet_rnn import ResNetRNN
from model_decoder import Decoder 
import time

def export_onnx(modelfile, onnxfile):
    checkpoint = torch.load(modelfile, map_location=lambda storage, loc: storage)
    print("checkpoint's type: {}".format(type(checkpoint)))
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            print("state_dict in checkpoint.")
            checkpoint = checkpoint['state_dict']
        if 'model' in checkpoint:
            print("model in checkpoint.")
            checkpoint = checkpoint['model']
    model = Decoder(odim=6532,
                    attention_dim=256,
                    attention_heads=4,
                    linear_units=2048,
                    num_blocks=6,
                    dropout_rate=0.1,
                    positional_dropout_rate=0.1,
                    self_attention_dropout_rate=0.0,
                    src_attention_dropout_rate=0.0)
    isDataParallel = False
    for i in checkpoint.keys():
        if i.startswith('module.'):
            isDataParallel = True 
            break
    print("isDataParallel: {}".format(isDataParallel))
    if isDataParallel:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    if isDataParallel:
        model = model.module
    
    model.cuda()
    model.eval()

    input_0 = torch.LongTensor([6531, 115, 336, 3150, 81])
    input_0 = input_0.unsqueeze(0)

    dummy_input_0 = Variable(input_0).cuda()
    dummy_input_1 = Variable(torch.randn(396, 256)).cuda()

    #ret = model(dummy_input_0, dummy_input_)
    #print("ret, shape: {s}".format(s=ret.shape))

    print("load model success. ")

    #dynamic_axes = {'input': {0 : '-1'}, 'output': {0 : '-1'}}
    dynamic_axes = {'input0': {1 : '-1'}, 'input1': {0 : '-1'}}
    #dummy_input = Variable(torch.randn(1, 3, 224, 224)) # nchw

    #torch.onnx.export(model, (dummy_input_0, dummy_input_1), onnxfile, verbose=True, \
    #                  opset_version=10, aten=False, \
    #                  input_names=('input0', 'input1'),
    #                  output_names=('output',),
    #                  dynamic_axes=dynamic_axes)

    torch.onnx.export(model, (dummy_input_0, dummy_input_1), onnxfile, verbose=True, \
                      opset_version=10, aten=False, \
                      input_names=('input0', "input1"),
                      output_names=('output',),
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                      keep_initializers_as_inputs=True)


def check_onnx_model(onnx_model_file):
    import onnx
    onnx_model = onnx.load(onnx_model_file)
    onnx.checker.check_model(onnx_model)
    print("===> Passed.")


def modify_onnx_batch_size(model_file, new_model_file):
    import onnx
    onnx_model = onnx.load(model_file)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'
    onnx.save(onnx_model, new_model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument("command", help="command to do", choices=["p2o", "o2t", "pth"])
    parser.add_argument('-b', '--batch-size', type=int, default=1, metavar='N', help="batch size.")
    parser.add_argument('-m', '--model-file', type=str, default="image-CNN-pt-vulgar.model.float32", \
                         metavar='N', help="model file.")

    args = parser.parse_args()
    if args.command == "p2o":
        ori_model_file = args.model_file
        onnx_model_file = ori_model_file + ".v12_param_unsqueeze.onnx"
        export_onnx(ori_model_file, onnx_model_file)

