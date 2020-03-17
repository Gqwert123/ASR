decoder_base4.pth.v5_2_param.onnx is Asr decoder model from espnet https://github.com/espnet/espnet.
input0's shape is [-1], variable size
input1's shape is [-1, 256], first dimension has variable size.


code:

decoder network: 
model_decoder.py  model_part.py utils.py

script which export pytorch model to onnx model:
pytorch2onnx.py

env:
python 3.6.5
pytorch 1.3.0
