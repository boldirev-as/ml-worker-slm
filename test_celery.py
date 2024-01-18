import time
from io import BytesIO

import cv2
import numpy as np
from ultralytics import YOLO

from tasks import add
from utils import processing, get_svg_data, get_defects_info

# model = YOLO('yolo_model_segm.pt')  # load a custom model
model = YOLO("models/yolo_model_segm.engine", task="segment")
# t = time.time()
img = cv2.imread('all_printers/printer/print4_01251-recoat.jpeg')
img = processing(img)

results = model.predict([img], imgsz=1024)
# results = model(img)
# error_ratio, annotated_frame = get_defects_info(results, svg, img)
# np_annotated_frame = np.array(annotated_frame.convert('RGB'))

# img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
#
# prev_img = cv2.imread('printer/00442-recoat.jpg')
# prev_img_bytes = cv2.imencode('.jpg', prev_img)[1].tobytes()
#
# svg_array = get_svg_data("printer/00296.svg")
# svg_bytes = svg_array.tobytes()
#
# result = add.delay(img_bytes, prev_img_bytes, svg_bytes, svg_array.shape)
# result = result.get(propagate=False)
# img = result['visualizations'][0]
# nparr = np.frombuffer(img, np.uint8)
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# cv2.imwrite('output.jpg', img)

# print(result['alerts'])

# img = img_bytes
# svg = svg_bytes
#
#
# def detect_defected_recoat(img, svg):
#
#     results = model(img)
#     print("only yolo", time.time() - t)
#
#     annotated_frame = get_defects_info(results, svg, img)
#     np_annotated_frame = np.array(annotated_frame.convert('RGB'))
#
#     return {
#         'visualizations': np_annotated_frame,
#         'alerts': []
#     }
#
#
# nparr = np.frombuffer(img, np.uint8)
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
# svg = np.frombuffer(svg, dtype=bool)
# svg = svg.reshape(svg_array.shape)
#
# print('SEND and LOAD', time.time() - t)
#
# img_preprocessed = processing(img)
# inference_funcs = [detect_defected_recoat]
#
# print('PREPROCESS', time.time() - t)
#
# server_response = {'visualizations': [],
#                    'alerts': []}
# for inference_func in inference_funcs:
#     func_response = inference_func(img_preprocessed.copy(), svg.copy())
#     if func_response['visualizations'] is not None:
#         img = func_response['visualizations']
#         img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
#         server_response['visualizations'].append(
#             img_bytes
#         )
#     if func_response['alerts'] is not None:
#         server_response['alerts'].append(func_response['alerts'])
#
# print('whole_PREDICT', time.time() - t)
#
# img = server_response['visualizations'][0]
# nparr = np.frombuffer(img, np.uint8)
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# cv2.imwrite('output.jpg', img)
#
# print(time.time() - t)
