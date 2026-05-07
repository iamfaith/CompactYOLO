import cv2
import numpy as np
from pathlib import Path

from typing import List, Tuple
from time import time
from models import YOLOv5_new as YOLOv5
from utils.general import check_img_size, scale_boxes, draw_detections, colors, increment_path, LoadMedia


img_size = [640, 640]

weights = 'base/best1024.onnx'

project = "test"
conf_thres = 0.1
iou_thres = 0.45
max_det = 1000

save_dir = increment_path(Path(project))
save_dir.mkdir(parents=True, exist_ok=True)


model = YOLOv5(weights, conf_thres, iou_thres, max_det, class_id = [0], verbose=True)
model.warmup(imgsz=(3, img_size[0], img_size[1]))
img_size = check_img_size(img_size, s=max(model.stride) if isinstance(model.stride, list) else model.stride)  # check img_size
print(img_size)



def inference(source, write_images=True):
    dataset = LoadMedia(source, img_size=img_size)
    for resized_image, original_image, status in dataset:
        image = resized_image.transpose(2, 0, 1)  # Convert from HWC -> CHW
        image = image[::-1]  # Convert BGR to RGB
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0  # Normalize the input
    
        start = time()
        # Model Inference
        boxes, scores, class_ids = model(image)
        end = time()
        inference_time = end - start
        # print(f"Total Time: {inference_time * 1000:.2f} ms")


        start = time()
        if len(boxes) > 0:
            # Scale bounding boxes to original image size
            boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

            # Draw bunding boxes
            for box, score, class_id in zip(boxes, scores, class_ids):
                draw_detections(original_image, box, score, model.names[int(class_id)], colors(int(class_id)))

            # Print results
            for c in np.unique(class_ids):
                n = (class_ids == c).sum()  # detections per class
                status += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

        
        # if view_img:
        #     # Display the image with detections
        #     cv2.imshow('Webcam Inference', original_image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        #         break

        # print(status)
        # print(f"postprocess Time: {(time() - start) * 1000:.2f} ms")
        if dataset.type == "image" and write_images:
            save_path = str(save_dir / f"frame_{dataset.frame:04d}.jpg")
            print(save_path)
            cv2.imwrite(save_path, original_image)
        return inference_time


from glob import glob

times = []

for img_path in glob('/home/faith/val/*.jpg'):
    inference_time = inference(img_path, write_images=False)
    times.append(inference_time)
    print(f"Processing {img_path} Total Time: {inference_time * 1000:.2f} ms")
    # break


if times:
    avg_time = np.mean(times)
    print(f"Average inference time: {avg_time * 1000:.2f} ms over {len(times)} images")
  
