from roboflow import Roboflow
# rf = Roboflow(api_key="wHRoGYzNQBOTzr9p0oBT")
# project = rf.workspace("trainingmodel-nbhhi").project("fish_detected")
# dataset = project.version("2").download("yolov5")


# import cv2
# import base64
# import numpy as np
# from ultralytics import YOLO
# # Carrega um modelo YOLOv5n pré-treinado no COCO
# modelo = YOLO('yolov5n.pt')

# # Mostra informações do modelo (opcional)
# modelo.info()

# # Treina o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
# resultados = modelo.train(data='fish_detected-2/data.yaml', epochs=4, imgsz=640)

# # Executa a inferência com o modelo YOLOv5n na imagem 'bus.jpg'
# resultados = modelo('2023.jpg')
# print(resultados)


from inference import get_roboflow_model
import supervision as sv
import cv2

# from inference.models.utils import get_model

# model = get_model(model_id="...", api_key="YOUR ROBOFLOW API KEY")

# image_file = "fish_1.jpg"
# image = cv2.imread(image_file)

# model = get_roboflow_model(model_id="fish_detected/2",api_key="wHRoGYzNQBOTzr9p0oBT")

# results = model.infer(image)

# detections = sv.Detections.from_roboflow(results[0].dict(by_alias=True, exclude_none=True))

# print(len(detections)," ready.")
# if len(detections) == 10:
#   print("10 screws counted. Package ready to move on.")
# else:
#   print(len(detections), "screws counted. Package is not ready.")


# from inference_sdk import InferenceHTTPClient

# # create an inference client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="wHRoGYzNQBOTzr9p0oBT"
# )

# # run inference on a local image
# print(CLIENT.infer(
#     "test1.jpeg", 
#     model_id="fish_detected/2"
# ))


import cv2
import base64
import numpy as np
import requests
import time 


# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
ROBOFLOW_API_KEY = "wHRoGYzNQBOTzr9p0oBT"
#for my face detection project, the model id is: "face-detection-mik1i"
ROBOFLOW_MODEL_ID = "fish_detected"
#for my face detection project, the version number is: "18" for this inference example
MODEL_VERSION = "2"
#resize value - for my face detection project, v18, it is 640
ROBOFLOW_SIZE = 640
#path to image for inference, insert path between the empty quotes
img_path = "test1.jpeg"

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


class rob():
    def infer(self,img, ROBOFLOW_SIZE, upload_url):
        # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
        height, width, channels = img.shape
        scale = ROBOFLOW_SIZE / max(height, width)
        img = cv2.resize(img, (round(scale * width), round(scale * height)))

        # Encode image to base64 string
        retval, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer)

        # Get prediction from Roboflow Infer API
        resp = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }, stream=True).raw

        # Parse result image
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image

# while 1:
#     # On "q" keypress, exit
#     if(cv2.waitKey(1) == ord('q')):
#         break

#     # Synchronously get a prediction from the Roboflow Infer API
#     image = infer()
#     # And display the inference results
#     cv2.imshow('image', image)

video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_BUFFERSIZE, 3)
while video.isOpened():
    # On "q" keypress, exit

    ret, img = video.read()
    result = img.copy()
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break
    elif(key == ord('r')):
        result = rob().infer(img, ROBOFLOW_SIZE, upload_url)
        try:
            if(result != None):
                cv2.imwrite("test4.jpg", result)
            else: print("Vz")
        except:
            cv2.imwrite("test4.jpeg", result)

    # And display the inference results
    
  

    cv2.imshow('image', result)
# Release resources when finished
video.release()
cv2.destroyAllWindows()

# img_file = cv2.imread(img_path)

