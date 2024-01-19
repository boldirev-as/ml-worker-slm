from celery import Celery
import numpy as np
from utils import processing, get_defects_info
from ultralytics import YOLO
import cv2
import pickle
from skimage.metrics import structural_similarity as peak_signal_noise_ratio

app = Celery('tasks', broker='redis://redis-stack-server:6379', backend='redis://redis-stack-server:6379')
# app = Celery('tasks', broker='redis://46.45.33.28:22080', backend='redis://46.45.33.28:22080')

app.control.time_limit('tasks.evaluate_layer',
                       soft=1, hard=2, reply=True)

# Load a model for segmentation
# model = YOLO('models/yolo_model_segm.pt')
model = YOLO("models/yolo_model_segm.engine", task="segment")

# open dump model for classification wiper
with open('models/model_logisticRegression.pkl', 'rb') as f:
    classifier = pickle.load(f)


def classify_metal_absence(img: np.array, prev_img: np.array, svg: np.array) -> dict:
    """
    Function for classification absence of metal material on layer (recoat)

    :param img: current layer image recoat
    :param prev_img: previous layer image reocat
    :param svg: svg matrix (bool mask) of current layer
    :return: dict with key alerts for information about issues (metric - similarity)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

    comparing = peak_signal_noise_ratio(img, prev_img)
    similarity_metric = classifier.predict_proba([[comparing]])[0][0]

    alerts = []
    if similarity_metric > 0.5:
        alerts.append({
            'value': similarity_metric,
            'info': 'There is no enough metal in SLM container',
            'error_type': 'METAL_ABSENCE'
        })

    return {
        'visualizations': None,
        'alerts': alerts
    }


def detect_defected_wiper(img: np.array, prev_img: np.array, svg: np.array) -> dict:
    """
    Function for segmentation wiper defects using recoat stage image

    :param prev_img: previous layer matrix (recoat)
    :param img: image matrix (recoat)
    :param svg: svg matrix (bool mask)
    :return: dict with key visualizations for image and alerts
    for information about issues (metric - error rate)
    """

    results = model.predict([img], imgsz=1024)
    # results = model(img)
    error_ratio, annotated_frame = get_defects_info(results, svg, img)
    np_annotated_frame = np.array(annotated_frame.convert('RGB'))

    alerts = []
    if results[0].masks is not None:
        alerts.append({
            'value': error_ratio,
            'info': 'Wiper defected and can affect the result of SLM',
            'error_type': 'WIPER_DEFECTED'
        })

    return {
        'visualizations': np_annotated_frame,
        'alerts': alerts
    }


@app.task
def evaluate_layer(recoat_img: bytes, previous_recoat_img: bytes, svg: bytes, shape: (int, int)) -> dict:
    """
    celery task for searching deffects in one layer, using svg file, scan and recoat images

    :param recoat_img: image from recoat stage in current layer (cv2 bytes)
    :param previous_recoat_img: image from recoat stage in previous layer (cv2 bytes)
    :param svg: svg file for current layer (np mask bytes)
    :param shape: shape of input images (tuple height and width)
    :return: dict file with images for visualization defects and
    a list with alerts + info
    """

    # Load all files from bytes format
    nparr = np.frombuffer(recoat_img, np.uint8)
    recoat_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    nparr = np.frombuffer(previous_recoat_img, np.uint8)
    previous_recoat_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    svg = np.frombuffer(svg, dtype=bool)
    svg = svg.reshape(shape)

    # Preprocess images to grayscale
    img_preprocessed = processing(recoat_img)
    previous_img_preprocessed = processing(previous_recoat_img)

    # modules for detection defects
    inference_funcs = [detect_defected_wiper, classify_metal_absence]

    server_response = {'visualizations': [],
                       'alerts': []}
    for inference_func in inference_funcs:
        # launch new module
        func_response = inference_func(img_preprocessed.copy(), previous_img_preprocessed.copy(), svg.copy())

        # if there is something wrong, save to server response
        if func_response['visualizations'] is not None:
            recoat_img = func_response['visualizations']
            img_bytes = cv2.imencode('.jpg', recoat_img)[1].tobytes()
            server_response['visualizations'].append(
                img_bytes
            )
        if func_response['alerts'] is not None:
            server_response['alerts'].extend(func_response['alerts'])

    return server_response
