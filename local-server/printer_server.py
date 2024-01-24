import time

import cv2
from celery import Celery
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import configparser

import numpy as np

from requests_for_backend import setup_layer, add_photos_to_layer, create_new_project
from utils import get_svg_data, processing

config = configparser.ConfigParser()
config.read("settings.ini")

# celery_client = Celery('tasks', broker='redis://0.0.0.0:8010',
#                        backend='redis://0.0.0.0:8010')
celery_client = Celery('tasks', broker='redis://46.45.33.28:22080',
                       backend='redis://46.45.33.28:22080')

CURRENT_PRINTER_DETAILS = {
    'PRINTER_UID': '213213',
    'PROJECT_ID': '65b0053215cc3ebf6d57c835'
}

REASON_TO_ID = {
    'WIPER_DEFECTED': 0,
    'METAL_ABSENCE': 1,
    'LAZER_INSTANCE': 2
}

app = FastAPI()


@app.get("/start_processing/")
def start(project_name: str, layer_number: int, svg_path: str, img_recoat_path: str, img_scan_path: str,
          img_recoat_prev_path: str):
    if layer_number == 1:
        CURRENT_PRINTER_DETAILS['PROJECT_ID'] = create_new_project(3000)

    if layer_number < 1:
        return {"warns": [], "project_id": CURRENT_PRINTER_DETAILS['PROJECT_ID'], "order": layer_number,
                "printer_uid": "", "before_melting_image": "",
                "after_melting_image": "",
                "svg_image": "", "id": ""}

    t = time.time()

    img = cv2.imread(img_recoat_path)
    img = processing(img)
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()

    after_melting_img = cv2.imread(img_scan_path)
    after_melting_img_processed = processing(after_melting_img)
    after_metling_img_flipped = np.rot90(np.fliplr(after_melting_img_processed), k=3)
    after_metling_img_flipped_bytes = cv2.imencode('.jpg', after_metling_img_flipped)[1].tobytes()
    # after_melting_img_bytes = cv2.imencode('.jpg', after_melting_img)[1].tobytes()

    prev_img = cv2.imread(img_recoat_prev_path)
    prev_img = processing(prev_img)
    prev_img_bytes = cv2.imencode('.jpg', prev_img)[1].tobytes()

    print('IMAGES PROCESSED', time.time() - t)
    t = time.time()

    svg_array, svg_png_bytes = get_svg_data(svg_path)
    svg_array = svg_array.astype('uint8')
    svg_bytes = cv2.imencode('.jpg', svg_array)[1].tobytes()

    print('SHAPES', img.shape, svg_array.shape)

    print('SVG DATA GOT', time.time() - t)

    t = time.time()

    task = celery_client.send_task('tasks.evaluate_layer',
                                   (img_bytes, prev_img_bytes, svg_bytes))

    try:
        result = task.get()
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Oops! Celery did something: {e}"},
        )

    print("DEFECTS COLLECTED", time.time() - t)

    t = time.time()
    if len(result['visualizations']) != 0:
        img_bytes = result['visualizations'][0]
        img_viz_array = np.frombuffer(img_bytes, np.uint8)
        img_viz = cv2.imdecode(img_viz_array, cv2.IMREAD_COLOR)
    else:
        img_viz = img

    img_flipped = np.rot90(np.fliplr(img_viz), k=3)
    img_flipped_bytes = cv2.imencode('.jpg', img_flipped)[1].tobytes()

    # SEND WORK TO back
    warns = []
    for alert in result['alerts']:
        warns.append({
            'reason': REASON_TO_ID[alert['error_type']],
            'rate': alert['value']
        })

    print(warns)

    print("WARNINGS PROCESSED", time.time() - t)

    t = time.time()

    # svg_bytes = open(svg_path, mode='rb').read()
    # svg_png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=800, output_height=800)

    layer_id = setup_layer(layer_number, CURRENT_PRINTER_DETAILS['PROJECT_ID'], warns, result['recommendation'])
    response = add_photos_to_layer(layer_id, img_flipped_bytes, after_metling_img_flipped_bytes,
                                   svg_png_bytes)

    print('image processed CARIO SVG + SEND', time.time() - t, response.text)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.json(),
    )
