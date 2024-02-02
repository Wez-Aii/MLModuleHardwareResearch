import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime.training import artifacts
from onnxruntime import InferenceSession
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import onnx
import io
# import netron
# import evaluate

# print("heelo war")

# # Pytorch class that we will use to generate the graphs.
# class MNISTNet(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MNISTNet, self).__init__()

#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(hidden_size, num_classes)

#     def forward(self, model_input):
#         out = self.fc1(model_input)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# # Create a MNISTNet instance.
# device = "cpu"
# batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
# pt_model = MNISTNet(input_size, hidden_size, output_size).to(device)


# # Generate a random input.
# model_inputs = (torch.randn(batch_size, input_size, device=device),)

# model_outputs = pt_model(*model_inputs)
# if isinstance(model_outputs, torch.Tensor):
#     model_outputs = [model_outputs]
    
# input_names = ["input"]
# output_names = ["output"]
# dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

# f = io.BytesIO()
# torch.onnx.export(
#     pt_model,
#     model_inputs,
#     f,
#     input_names=input_names,
#     output_names=output_names,
#     opset_version=14,
#     do_constant_folding=False,
#     training=torch.onnx.TrainingMode.TRAINING,
#     dynamic_axes=dynamic_axes,
#     export_params=True,
#     keep_initializers_as_inputs=False,
# )
# onnx_model = onnx.load_model_from_string(f.getvalue())
onnx_model = onnx.load('/workspaces/Wez/models/ssd_mobilenet_v1_coco_2018_01_28.onnx')
# onnx_model = InferenceSession('/workspaces/Wez/models/retinanet_rn34_1280x768_dummy.onnx')

# model_params = onnx_model.training_info
# print("model_params -", model_params)

# model_params = onnx_model.get_modelmeta()
# print("model_params -", model_params)
requires_grad = []
frozen_params = []
# for param in model_params:
#     if param.trainable:
#         print(f"{param.name} requires gradients")
#         requires_grad.append(param.name)
#     else:
#         print(f"{param.name} does not require gradients")
#         frozen_params.append(param.name)

for initializer in onnx_model.graph.initializer:
    # Check for attributes or naming conventions indicating trainability
    if hasattr(initializer, "trainable") and initializer.trainable:
        print(f"{initializer.name} requires gradients")
        requires_grad.append(initializer.name)
    # If attributes are not available, infer from naming conventions
    elif "weights" in initializer.name or "bias" in initializer.name:
        print(f"{initializer.name} likely requires gradients")
        requires_grad.append(initializer.name)
    else:
        print(f"{initializer.name} likely does not require gradients")
        frozen_params.append(initializer.name)

print(len(requires_grad), "requires grad -", requires_grad)
# print(len(frozen_params), "frozen grad -", frozen_params)


# # model = torch.onnx.load('/workspaces/Wez/models/ssd_mobilenet_v1_coco_2018_01_28.onnx')

# # print(type(model))

# dummy_input = torch.randn((1, 3, 224, 224))

# # Export the ONNX model to a PyTorch model
# model = onnx.load('/workspaces/Wez/models/ssd_mobilenet_v1_coco_2018_01_28.onnx')
# torch.onnx.export(model, dummy_input, '/workspaces/Wez/models/ssd_mobilenet_v1_coco_2018_01_28.pth', verbose=True)

# # Load the PyTorch model
# pytorch_model = torch.load('path/to/pytorch_model.pth')
# pytorch_model.eval()

# print(type(pytorch_model))



# import onnx
# import onnxruntime.training.onnxblock as onnxblock
# from onnxruntime.training import artifacts

# # Define a custom loss block that takes in two inputs
# # and performs a weighted average of the losses from these
# # two inputs.
# class WeightedAverageLoss(onnxblock.Block):
#     def __init__(self):
#         self._loss1 = onnxblock.loss.MSELoss()
#         self._loss2 = onnxblock.loss.MSELoss()
#         self._w1 = onnxblock.blocks.Constant(0.4)
#         self._w2 = onnxblock.blocks.Constant(0.6)
#         self._add = onnxblock.blocks.Add()
#         self._mul = onnxblock.blocks.Mul()

#     def build(self, loss_input_name1, loss_input_name2, v1, v2):
#         # The build method defines how the block should be stacked on top of
#         # loss_input_name1 and loss_input_name2

#         # Returns weighted average of the two losses
#         return self._add(
#             self._mul(self._w1(), self._loss1(loss_input_name1, target_name="target1")),
#             self._mul(self._w2(), self._loss2(loss_input_name2, target_name="target2"))
#         )

# # Load the ONNX model
# model_path = "model.onnx"
# base_model = onnx.load('/workspaces/Wez/models/ssd_mobilenet_v1_coco_2018_01_28.onnx')

# # Specify the parameters that need their gradient computed
# requires_grad = ["weight1", "bias1", "weight2", "bias2"]

# # Specify the frozen parameters
# frozen_params = ["weight3", "bias3"]

# # Instantiate your custom loss
# my_custom_loss = WeightedAverageLoss()

# # Invoke generate_artifacts with the custom loss function and other parameters
# artifacts.generate_artifacts(
#     base_model,
#     requires_grad=requires_grad,
#     frozen_params=frozen_params,
#     loss=my_custom_loss,
#     optimizer=artifacts.OptimType.AdamW
# )


# Generate the training artifacts
artifacts.generate_artifacts(onnx_model,
                             requires_grad = requires_grad,
                             frozen_params = frozen_params,
                             loss = artifacts.LossType.CrossEntropyLoss,
                             optimizer = artifacts.OptimType.AdamW,
                             artifact_directory = "data")


# requires_grad = [name for name, param in pt_model.named_parameters() if param.requires_grad]

# frozen_params = [name for name, param in pt_model.named_parameters() if not param.requires_grad]

# artifacts.generate_artifacts(
#     onnx_model,
#     optimizer=artifacts.OptimType.AdamW,
#     loss=artifacts.LossType.CrossEntropyLoss,
#     requires_grad=requires_grad,
#     frozen_params=frozen_params,
#     artifact_directory="data",
#     additional_output_names=["output"])



# # Creating a class with a Loss function.
# class MNISTTrainingBlock(onnxblock.TrainingBlock):
#     def __init__(self):
#         super(MNISTTrainingBlock, self).__init__()
#         self.loss = onnxblock.loss.CrossEntropyLoss()

#     def build(self, output_name):
#         return self.loss(output_name), output_name
    


# # Build the onnx model with loss
# training_block = MNISTTrainingBlock()
# for param in onnx_model.graph.initializer:
#     print(param.name)
#     training_block.requires_grad(param.name, True)

# # Building training graph and eval graph.
# model_params = None
# with onnxblock.base(onnx_model):
#     _outputname = [output.name for output in onnx_model.graph.output]
#     _ = training_block(*_outputname)
#     training_model, eval_model = training_block.to_model_proto()
#     model_params = training_block.parameters()

# # Building the optimizer graph
# optimizer_block = onnxblock.optim.AdamW()
# with onnxblock.empty_base() as accessor:
#     _ = optimizer_block(model_params)
#     optimizer_model = optimizer_block.to_model_proto()



# # Saving all the files to use them later for the training.
# onnxblock.save_checkpoint(training_block.parameters(), "data/checkpoint")
# onnx.save(training_model, "data/training_model.onnx")
# onnx.save(optimizer_model, "data/optimizer_model.onnx")
# onnx.save(eval_model, "data/eval_model.onnx")




# batch_size = 64
# train_kwargs = {'batch_size': batch_size}
# test_batch_size = 1000
# test_kwargs = {'batch_size': test_batch_size}

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
#     ])

# dataset1 = datasets.MNIST('../data', train=True, download=True,
#                     transform=transform)
# dataset2 = datasets.MNIST('../data', train=False,
#                     transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
# test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)




# # Create checkpoint state.
# state = CheckpointState.load_checkpoint("data/checkpoint")

# # Create module.
# model = Module("data/training_model.onnx", state, "data/eval_model.onnx")

# # Create optimizer.
# optimizer = Optimizer("data/optimizer_model.onnx", model)




# # Util function to convert logits to predictions.
# def get_pred(logits):
#     return np.argmax(logits, axis=1)

# # Training Loop :
# def train(epoch):
#     model.train()
#     losses = []
#     for _, (data, target) in enumerate(train_loader):
#         forward_inputs = [data.reshape(len(data),784).numpy(),target.numpy().astype(np.int64)]
#         train_loss, _ = model(*forward_inputs)
#         optimizer.step()
#         model.lazy_reset_grad()
#         losses.append(train_loss)

#     print(f'Epoch: {epoch+1},Train Loss: {sum(losses)/len(losses):.4f}')

# # Test Loop :
# def test(epoch):
#     model.eval()
#     losses = []
#     metric = evaluate.load('accuracy')

#     for _, (data, target) in enumerate(train_loader):
#         forward_inputs = [data.reshape(len(data),784).numpy(),target.numpy().astype(np.int64)]
#         test_loss, logits = model(*forward_inputs)
#         metric.add_batch(references=target, predictions=get_pred(logits))
#         losses.append(test_loss)

#     metrics = metric.compute()
#     print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')



# for epoch in range(5):
#     train(epoch)
#     # test(epoch)

    


# model.export_model_for_inferencing("data/inference_model.onnx",["output"])
# session = InferenceSession('data/inference_model.onnx',providers=['CPUExecutionProvider'])



# # getting one example from test list to try inference.
# data = next(iter(test_loader))[0][0]

# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name 
# output = session.run([output_name], {input_name: data.reshape(1,784).numpy()})

# # plotting the picture
# plt.imshow(data[0], cmap='gray')
# plt.show()

# print("Predicted Label : ",get_pred(output[0]))