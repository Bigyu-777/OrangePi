# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

CLASSES = ['car', 'pickup', 'camping','truck', 'other', 'tractor', 'boat', 'van']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = InferSession(device_id=0, model_path="best_huawei.om")

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    if class_id >= len(CLASSES):
        print(f"[Warning] Invalid class_id {class_id}")
        return
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def preprocess(img, size=(1024, 1024)):
    img_resized = cv2.resize(img, size)
    blob = cv2.dnn.blobFromImage(img_resized, scalefactor=1/255.0, size=size, swapRB=True)
    return blob, img_resized

def infer_and_draw(img_rgb, img_ir):
    blob_rgb, resized_rgb = preprocess(img_rgb)
    blob_ir, _ = preprocess(img_ir)

    scale_x = img_rgb.shape[1] / 1024
    scale_y = img_rgb.shape[0] / 1024

    start_time = time.time()
    outputs = model.infer(feeds=[blob_rgb, blob_ir], mode="static")
    end_time = time.time()
    inference_time = end_time - start_time

    outputs = outputs[0]  # Assume outputs shape: [1, N, 85]
    if len(outputs.shape) == 3:
        outputs = outputs[0]  # shape: [N, 85]

    print(f"[Debug] Output shape: {outputs.shape}")
    rows = outputs.shape[0]

    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        box_data = outputs[i]
        obj_conf = box_data[4]
        class_scores = box_data[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        final_conf = obj_conf * class_conf
    
        if final_conf >= 0.25 and class_id < len(CLASSES):
            cx, cy, w, h = box_data[0], box_data[1], box_data[2], box_data[3]
            x = cx - 0.5 * w
            y = cy - 0.5 * h
            boxes.append([x, y, w, h])
            scores.append(float(final_conf))
            class_ids.append(class_id)


    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    print(f"[Debug] Boxes before NMS: {len(boxes)}")
    print(f"[Debug] Boxes after NMS: {len(indices)}")

    for i in indices:
        index = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[index]
        draw_bounding_box(
            img_rgb,
            class_ids[index],
            scores[index],
            round(box[0] * scale_x),
            round(box[1] * scale_y),
            round((box[0] + box[2]) * scale_x),
            round((box[1] + box[3]) * scale_y),
        )

    return img_rgb, inference_time

rgb_folder = '/home/HwHiAiUser/multispetral/val/RGB/images/'
ir_folder = '/home/HwHiAiUser/multispetral/val/IR/images/'
output_folder = 'output_dualmodal'
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
total_inference_time = 0.0

for filename in os.listdir(rgb_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        rgb_path = os.path.join(rgb_folder, filename)
        ir_path = os.path.join(ir_folder, filename)

        if not os.path.exists(ir_path):
            continue

        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)

        if img_rgb is not None and img_ir is not None:
            result_img, inference_time = infer_and_draw(img_rgb, img_ir)

            total_inference_time += inference_time
            frame_count += 1

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_img)

if frame_count > 0:
    avg_fps = frame_count / total_inference_time
    print(f"Average FPS: {avg_fps:.2f}")
