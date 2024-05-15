import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import transforms as T
import torchvision.transforms as transforms
import time
from myutil import *
from model import RPN,TwoStageDetector

class_to_idx = {"mouse":0, 
                "keyboard":1,
                "laptop":2,
                "cell phone":3
                }
idx_to_class = {i:c for c, i in class_to_idx.items()}

# Load your Faster R-CNN model

def load_model():
    # Instantiate your Faster R-CNN model and load trained weights
    model = TwoStageDetector()  # Example: Replace RPN with your actual model class
    model.load_weights("twostage.pth")
    # Load trained weights if necessary
    return model

# Detect objects in a frame
def detect_objects(frame, model,w,h,thr,nms_thr):
    # Perform object detection using your model
    # Example: detections = model.inference(frame)
    # valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])

    final_proposals, final_conf_scores, final_class = model.inference(frame,thresh=thr, nms_thresh=nms_thr)
    # print(final_proposals,final_conf_scores, final_class)
    final_proposals = torch.cat(final_proposals)
    final_class = torch.cat(final_class)
    final_conf_scores = torch.cat(final_conf_scores)
    # print(final_conf_scores,final_class,final_proposals)
    final_all = torch.cat((final_proposals, final_class, final_conf_scores),dim=1).cpu()
    resized_proposals = coord_trans(final_all, h, w)
    return frame,resized_proposals
import cv2
import numpy as np

def draw_boxes(frame, boxes, idx_to_class):
    for box in boxes:
        xmin, ymin, xmax, ymax, class_idx, _ = box
        class_label = idx_to_class[class_idx.item()]
        color = (0, 255, 0)  # Green color for bounding boxes
        thickness = 2  # Thickness of bounding box lines

        # Draw bounding box rectangle
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

        # Put class label text above the bounding box
        label = f"{class_label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = int(xmin)
        text_y = int(ymin - text_size[1] - 2)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, color, font_thickness)

    return frame

# Modify your main function to draw bounding boxes directly on the camera frames
def main(video_path=None):
    fps=0.0
    # Load the Faster R-CNN model
    model = load_model().eval()

    # Open the default camera (index 0)
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        exit()

    # Main loop to continuously capture frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        t1=time.time()
        h = frame.shape[0]  # Height of the image
        w = frame.shape[1]  # Width of the image

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Couldn't capture frame.")
            break
        preprocess = transforms.Compose([
                        transforms.ToPILImage(),  # Convert tensor to PIL Image
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
        # Detect objects in the frame
        transformed_frame = preprocess(frame)
        detframe, box = detect_objects(transformed_frame, model, h, w, thr=0.6, nms_thr=0.3)
        fps = ( fps + (1./(time.time()-t1)) ) / 2

        # Draw bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame, box, idx_to_class)
        print("fps= %.2f"%(fps))

        # Display the frame with bounding boxes
        cv2.imshow("Object Detection", frame_with_boxes)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    video_path="IMG_4426.mov"
    main("IMG_4426.mov")
