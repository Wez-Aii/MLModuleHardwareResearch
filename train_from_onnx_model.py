import onnxruntime
from onnxruntime import make_pytorch_model

onnx_model = onnxruntime.InferenceSession("/workspaces/Wez/models/fruit/ssd-mobilenet.onnx")

pt_model = make_pytorch_model(onnx_model, strict=False)

print("hello")