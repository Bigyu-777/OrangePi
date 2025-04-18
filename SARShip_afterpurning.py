# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

# 类别
CLASSES = ['ship']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = InferSession(device_id=0, model_path="best_huawei.om")

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(original_image):
    [height, width, _] = original_image.shape
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    # 只计算推理时间
    start_time = time.time()
    outputs = model.infer(feeds=[blob], mode="static")
    end_time = time.time()
    inference_time = end_time - start_time

    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - 0.5 * outputs[0][i][2],
                outputs[0][i][1] - 0.5 * outputs[0][i][3],
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    return original_image, inference_time  # 返回处理后图像和推理时间


# 主逻辑
input_image_folder = '/home/HwHiAiUser/val/images'
out_image_folder = 'output'

# 确保输出文件夹存在
os.makedirs(out_image_folder, exist_ok=True)

frame_count = 0
total_inference_time = 0.0

for filename in os.listdir(input_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_image_folder, filename)
        image = cv2.imread(image_path)

        if image is not None:
            processed_image, inference_time = main(image)

            total_inference_time += inference_time
            frame_count += 1

            # ✅ 保存处理后的图像
            output_path = os.path.join(out_image_folder, filename)
            cv2.imwrite(output_path, processed_image)

# 输出平均推理FPS
if frame_count > 0:
    average_fps = frame_count / total_inference_time
    print(f"Average FPS (仅推理): {average_fps:.2f}")
