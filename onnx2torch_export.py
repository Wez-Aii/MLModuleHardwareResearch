import onnx
import torch
from onnx2torch import convert

# Path to ONNX model
""" Error on convert """
# onnx_model_path = '/workspaces/Wez/models/resnet50_v1.onnx'
# onnx_model_path = '/workspaces/Wez/models/resnext50_32x4d_fpn.onnx'
# onnx_model_path = '/workspaces/Wez/models/ssd_mobilenet_v1_coco_2018_01_28.onnx'

""" Success on convert """
# onnx_model_path = '/workspaces/Wez/models/retinanet_rn34_1280x768_dummy.onnx'
onnx_model_path = '/workspaces/Wez/models/resnet34-ssd1200.onnx'


# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

onnx.helper.printable_graph(onnx_model.graph)  

print(type(onnx_model))

torch_model_2 = convert(onnx_model)

print(type(torch_model_2))

torch.save(torch_model_2, "/workspaces/Wez/models/testing.pth")

model = torch.load("/workspaces/Wez/models/testing.pth")
model.eval()