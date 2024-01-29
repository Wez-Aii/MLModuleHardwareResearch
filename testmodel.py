import onnxruntime as rt
import cv2
import numpy as np

import onnx
import torch

from vision.utils import box_utils

def format_onnx_result(result):
	boxes = result[1]  # Assumes output is a list of bounding boxes
	scores = result[0]
	# detections = np.squeeze(result[1])  # Assumes output is a list of bounding boxes
	print("boxes type -", type(boxes))
	print("boxes -", boxes[0])
	print("scores type -", type(scores))
	print("scores -", scores[0])

	boxes = torch.from_numpy(boxes)
	scores = torch.from_numpy(scores)

	boxes = boxes[0]
	scores = scores[0]

	boxes = boxes.to(torch.device("cpu"))
	scores = scores.to(torch.device("cpu"))
	picked_box_probs = []
	picked_labels = []

	for class_index in range(1, scores.size(1)):
		probs = scores[:, class_index]
		print("probs -",probs)
		mask = probs > 0.4
		print("mask -",mask)
		probs = probs[mask]
		print("class index -",class_index)
		if probs.size(0) == 0:
			continue
			
		subset_boxes = boxes[mask, :]
		box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
		box_probs = box_utils.nms(box_probs, None,
									score_threshold=0.4,
									iou_threshold=0.45,
									sigma=0.5,
									top_k=10,
									candidate_size=200)
		picked_box_probs.append(box_probs)
		picked_labels.extend([class_index] * box_probs.size(0))
		
	if not picked_box_probs:
		print("not picked box probs")
		# return torch.tensor([]), torch.tensor([]), torch.tensor([])
		
	picked_box_probs = torch.cat(picked_box_probs)
	height, width, _ = image.shape
	picked_box_probs[:, 0] *= width
	picked_box_probs[:, 1] *= height
	picked_box_probs[:, 2] *= width
	picked_box_probs[:, 3] *= height

	return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


onnx_model = onnx.load("/workspaces/Wez/models/fruit/ssd-mobilenet.onnx")
onnx.checker.check_model(onnx_model)

# Load the model
model_path = "/workspaces/Wez/models/fruit/ssd-mobilenet.onnx"
session = rt.InferenceSession(model_path)

# Prepare input image
# image_path = "/workspaces/Wez/DLtestImages/istockphoto-637563258-612x612.jpg"
# image_path = "/workspaces/Wez/data/fruit/validation/02aeb6528711637a.jpg"
# image_path = "/workspaces/Wez/DLtestImages/360_F_463301545_BclWPd5elIS5T802eIweMpiuj3S2BMv9.jpg"
image_path = "/workspaces/Wez/DLtestImages/istockphoto-689765520-612x612.jpg"
# image_path = "/workspaces/Wez/DLtestImages/How-to-Make-Caramelized-Apples-and-Onions.jpg"
image = cv2.imread(image_path)
_copy_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# _copy_image = image.copy()

# Preprocess _copy_image (adjust based on your model's requirements)
input_shape = (1, 3, 300, 300)  # Example input shape
print(input_shape[2:4])
print(_copy_image.shape)
# _copy_image = cv2.resize(_copy_image, (300,300))
_copy_image = cv2.resize(_copy_image, (300, 300), interpolation=cv2.INTER_AREA)
# Convert to 4-dimensional tensor with color channels first and batch dimension as the first axis
_copy_image = _copy_image.transpose((2, 0, 1))
_copy_image = _copy_image.astype(np.float32) / 255.0
_copy_image = np.expand_dims(_copy_image, axis=0)

# Get model input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
result = session.run(['scores', 'boxes'], {input_name: _copy_image})
# result = session.run([output_name], {input_name: _copy_image})

# Process detections (adjust based on your model's output format)
# print(result[0])
boxes, labels, probs = format_onnx_result(result)

print("---boxes----",boxes)

class_names = ["BACKGROUND",
"Apple",
"Banana",
"Grape",
"Orange",
"Pear",
"Pineapple",
"Strawberry",
"Watermelon"]

for i in range(boxes.size(0)):
    box = boxes[i, :]
    # print("box", box)
    # box = box.squeeze().tolist()
    # print("box", box)
    # print("box", box[0], box[1], box[2], box[3])
    # point_1 = box[0], box[1]
    # point_2 = box[2], box[3]
    # point_1 = int(box[0]), int(box[1])
    # point_2 = int(box[2]), int(box[3])
    # cv2.rectangle(orig_image, box[0], box[1], box[2], box[3], (255, 255, 0), 4)
    # cv2.rectangle(orig_image, point_1, point_2, (255, 255, 0), 4)
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "test.jpg"
cv2.imwrite(path, image)
print(f"Found {len(probs)} objects. The output image is {path}")

# Draw bounding boxes on the image
# for detection in detections:
#     print(f"detection - {detection}")
	
# # for i in range(detections.size(0)):
# # for detection in detections:
# #     print(f"detection - {detections[0,:]}")
#     xmin, ymin, xmax, ymax = detection
# # #     xmin, ymin, xmax, ymax, confidence, class_id = detection
#     label = "Fruit"  # Replace with actual class labels if available

#     cv2.rectangle(image, (int(xmin*1000), int(ymin*1000)), (int(xmax*1000), int(ymax*1000)), (255, 255, 0), 2)
#     cv2.putText(image, f"{label}: --", (int(xmin), int(ymin - 5)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#     # cv2.putText(image, f"{label}: {confidence:.2f}", (int(xmin), int(ymin - 5)),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     break

# # Display the image with bounding boxes
# cv2.putText(image, "just test", (100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
# cv2.imwrite("/workspaces/Wez/test.png", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
	
# threshold = 0.5
# detected_classes = []
# for j in range(detections.shape[0]):
#     for class_index in range(detections.shape[1]):
#         score = detections[j, class_index]
#         if score > threshold:
#             detected_classes.append({
#                 'class_index': class_index,
#                 'class_name': f'Class_{class_index}',
#                 'score': score,
#                 'location': (j, class_index)
#             })

# # for i in range(detections.shape[0]):
# #     for j in range(detections.shape[1]):
# #         for class_index in range(detections.shape[2]):
# #             score = detections[i, j, class_index]
# #             if score > threshold:
# #                 detected_classes.append({
# #                     'class_index': class_index,
# #                     'class_name': f'Class_{class_index}',
# #                     'score': score,
# #                     'location': (i, j)
# #                 })

# # Print the detected classes
# for detection in detected_classes:
#     print(f"Class {detection['class_name']} detected with a confidence of {detection['score']} at location {detection['location']}.")