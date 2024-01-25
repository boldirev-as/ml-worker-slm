import onnxruntime
import scipy
from PIL import Image
from celery import Celery
import numpy as np
from torchvision import transforms

from utils import get_defects_info, cut_image_by_mask, processing
from ultralytics import YOLO
import cv2
import pickle
from skimage.metrics import peak_signal_noise_ratio

app = Celery('tasks', broker='redis://redis-stack-server:6379', backend='redis://redis-stack-server:6379')
# app = Celery('tasks', broker='redis://46.45.33.28:22080', backend='redis://46.45.33.28:22080')

app.control.time_limit('tasks.evaluate_layer',
                       soft=1, hard=2, reply=True)

# Load a model for segmentation
yolo_model = YOLO("models/yolo_model_segm.engine", task="segment")

# open dump model for classification wiper
with open('models/model_regression_cut.pkl', 'rb') as f:  # open dump model
    classifier = pickle.load(f)

ort_session = onnxruntime.InferenceSession("models/lazer_classifier.onnx")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def detect_lazer(img: np.array, prev_img: np.array, svg: np.array) -> dict:
    """
    detect lazer on recoat stage (no need to run other algorithms in terms of polluting image)
    :param img: current layer image recoat
    :param prev_img: previous layer image reocat
    :param svg: svg matrix (bool mask) of current layer
    :return: dict with key alerts for information about issues (probability)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    transformed_image = transform(img).unsqueeze(0)

    onnxruntime_outputs = ort_session.run(None, {'arg0': transformed_image.numpy()})

    error_prob = scipy.special.softmax(onnxruntime_outputs[0])[0][1]

    alerts = []
    if error_prob > 0.5:
        alerts.append({
            'value': error_prob,
            'info': 'Lazer polluted photo',
            'error_type': 'LAZER_INSTANCE'
        })

    return {
        'visualizations': None,
        'alerts': alerts
    }


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

    cut_img = cut_image_by_mask(img, svg)
    cut_prev_img = cut_image_by_mask(prev_img, svg)

    comparing = peak_signal_noise_ratio(cut_img, cut_prev_img)
    similarity_metric = classifier.predict([[comparing]])[0]

    alerts = []
    recommendation = ''
    if similarity_metric > 0.5:
        alerts.append({
            'value': float(similarity_metric),
            'info': 'There is no enough metal in SLM container',
            'error_type': 'METAL_ABSENCE'
        })
        recommendation = 'metal_absence_stop'

    return {
        'visualizations': None,
        'alerts': alerts,
        'recommendation': recommendation
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

    results = yolo_model.predict([img], imgsz=1024)
    error_ratio, annotated_frame = get_defects_info(results, svg, img)
    np_annotated_frame = np.array(annotated_frame.convert('RGB'))

    alerts = []
    recommendation = ''
    if results[0].masks is not None:
        alerts.append({
            'value': error_ratio,
            'info': 'Wiper defected and can affect the result of SLM',
            'error_type': 'WIPER_DEFECTED'
        })
        recommendation = 'reslice_stop' if error_ratio > 0 else 'ignore'

    return {
        'visualizations': np_annotated_frame,
        'alerts': alerts,
        'recommendation': recommendation
    }


@app.task
def evaluate_layer(recoat_img: bytes, previous_recoat_img: bytes, svg: bytes) -> dict:
    """
    celery task for searching deffects in one layer, using svg file, scan and recoat images

    :param recoat_img: image from recoat stage in current layer (cv2 bytes)
    :param previous_recoat_img: image from recoat stage in previous layer (cv2 bytes)
    :param svg: svg file for current layer (np mask bytes)
    :return: dict file with images for visualization defects and
    a list with alerts + info
    """

    # Load all files from bytes format
    nparr = np.frombuffer(recoat_img, np.uint8)
    recoat_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    recoat_processed_img = processing(recoat_img)

    nparr = np.frombuffer(previous_recoat_img, np.uint8)
    previous_recoat_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    previous_recoat_processed_img = processing(previous_recoat_img)

    # decode svg layer to bin mask
    nparr = np.frombuffer(svg, np.uint8)
    svg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    svg = cv2.resize(svg, (1024, 1024))
    svg = svg[:, :, 0].reshape((1024, 1024)).astype(bool)

    # modules for detection defects
    inference_funcs = [
        (detect_lazer, {'use_preprocessed': False, 'add_to_response': False}),
        (classify_metal_absence, {}),
        (detect_defected_wiper, {})
    ]

    server_response = {
        'visualizations': [],
        'alerts': [],
        'recommendation': ''
    }
    for inference_func, func_metadata in inference_funcs:
        # launch new module
        if func_metadata.get('use_preprocessed', True) is True:
            func_response = inference_func(
                recoat_processed_img.copy(), previous_recoat_processed_img.copy(), svg.copy())
        else:
            func_response = inference_func(
                recoat_img.copy(), previous_recoat_img.copy(), svg.copy())

        # if there is something wrong, save to server response
        if func_response['visualizations'] is not None:
            recoat_img = func_response['visualizations']
            img_bytes = cv2.imencode('.jpg', recoat_img)[1].tobytes()
            server_response['visualizations'].append(
                img_bytes
            )

        if len(func_response['alerts']) != 0:
            if func_metadata.get('add_to_response', True) is True:
                server_response['alerts'].extend(func_response['alerts'])
                server_response['recommendation'] = func_response['recommendation']
            # in case of lazer (or other stuff) on photo we are not able to launch
            # other algorithms for defect spotting
            break

    return server_response
