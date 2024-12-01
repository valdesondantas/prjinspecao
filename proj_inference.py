"""
Desenvolvedor: Valdeson  Dantas de Souza
Data: 19/11/2024

"""

from roboflow import Roboflow
from numpy import mean
import supervision as sv
import cv2
import numpy as np


fpsArray = []
averageFPS = 0
# font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
org = (25, 25)
fontScale = 1
color = (255, 0, 0)
thickness = 1

pixel_ratio_array = []
averagePR = 0

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
ROBOFLOW_API_KEY = "ztSz6BNhvjL0DlLpb4M3"
# ROBOFLOW_API_KEY = "wHRoGYzNQBOTzr9p0oBT"
#for my face detection project, the model id is: "face-detection-mik1i"
ROBOFLOW_MODEL_ID = "fish_detected-rdlfv"
#for my face detection project, the version number is: "18" for this inference example
MODEL_VERSION = "3"
#resize value - for my face detection project, v18, it is 640
ROBOFLOW_SIZE = 640
#path to image for inference, insert path between the empty quotes
img_path = "test5.jpg"
results={}
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL_ID, "/",
    MODEL_VERSION,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&labels=on",
    "&stroke=2"
])

print(upload_url)

# Infer via the Roboflow Hosted Inference API and return the result
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_MODEL_ID)
model = project.version(MODEL_VERSION).model



model.confidence = 14
model.overlap = 10
model.stroke = 2

print(model)


job_id, signed_url,expire_time = model.predict_video(
    "videos/fish.mp4",
    fps=5,
    prediction_type="batch-video",
)


results = model.poll_until_video_results(job_id,41,14)


box_mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
cap = cv2.VideoCapture("videos/fish.mp4")
frame_rate = cap.get(cv2.CAP_PROP_FPS)

cap.release()

def annotate_frame(frame: np.ndarray, frame_number: int) -> np.ndarray:
    try:
        time_offset = frame_number / frame_rate
        closest_time_offset = min(results['time_offset'], key=lambda t: abs(t - time_offset))
        index = results['time_offset'].index(closest_time_offset)
        detection_data = results["fish_detected-rdlfv"][index]

        roboflow_format = {
            "predictions": detection_data['predictions'],
            "image": {"width": frame.shape[1], "height": frame.shape[0]
            }
        }
       
       
       
        detections = sv.Detections.from_inference(roboflow_format)
        # detections = detections[detections.confidence > model.confidence]
        detections = tracker.update_with_detections(detections)



        # labels = [f"#{tracker_id} {model.name[class_id]} {confidence:0.2f}"
        #     for _, _, confidence, class_id, tracker_id
        #     in detections
        #     ]
        labels = [pred['class'] for pred in detection_data['predictions']]
        predictions = detection_data["predictions"]
        for object1 in predictions:
            
            # print(object.json())
            object_JSON = object1

            object_class = str(object_JSON['class'])
            object_class_text_size = cv2.getTextSize(object_class, font, fontScale, thickness)
            print("CLASS: " + object_class)
            object_confidence = str(round(object_JSON['confidence']*100 , 2)) + "%"
            print("CONFIDENCE: " + object_confidence)

            # pull bbox coordinate points
            x0 = object_JSON['x'] - object_JSON['width'] / 2
            y0 = object_JSON['y'] - object_JSON['height'] / 2
            x1 = object_JSON['x'] + object_JSON['width'] / 2
            y1 = object_JSON['y'] + object_JSON['height'] / 2
            box = (x0, y0, x1, y1)
            # print("Bounding Box Cordinates:" + st
         
        
            averagePR = mean(pixel_ratio_array)

              ## THIS IS WHERE THE PIXEL RATIO IS CREATED ##
            if object_class == "nematoda_tucunare":
            
                soda_inches = 4.83

                soda_height = object_JSON['height']

                pixel_to_inches = soda_height / soda_inches
                pixel_ratio_array.append(pixel_to_inches)
                # print(pixel_to_inches)

                object_Inches = soda_height / averagePR

                print("SODA INCHES: " + str(object_Inches))

                inches_ORG = (int(x0), int(y0+120))

                frame = cv2.putText(frame, 'cm: ' + str(object_Inches)[:4], inches_ORG, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)


    except (IndexError, KeyError, ValueError) as e:
        print(f"Exception in processing frame {frame_number}: {e}")
        detections = sv.Detections(xyxy=np.empty((0, 4)),
                                   confidence=np.empty(0),
                                   class_id=np.empty(0, dtype=int))
        labels = []

    

    annotated_frame = box_mask_annotator.annotate(frame.copy(), detections=detections)
    
    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return annotated_frame


sv.process_video(
   source_path="videos/fish.mp4",
   target_path="videos/output9.mp4",
   callback=annotate_frame
)