import cv2
from fastapi import FastAPI
import configparser

import numpy as np
from tasks import evaluate_layer
from requests_for_backend import setup_layer, add_photos_to_layer
from utils import get_svg_data

config = configparser.ConfigParser()
config.read("settings.ini")

REASON_TO_ID = {
    'WIPER_DEFECTED': 0,
    'METAL_ABSENCE': 1
}

app = FastAPI()


@app.get("/start_processing/")
def start(project_name: str, layer_number: int):
    if layer_number < 40:
        return '200'

    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    img = cv2.imread(f'{project_name}/{add_zeros}{layer_number}-recoat.jpg')
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()

    after_melting_img = cv2.imread(f'{project_name}/{add_zeros}{layer_number}-scan.jpg')
    after_melting_img_bytes = cv2.imencode('.jpg', after_melting_img)[1].tobytes()

    prev_img = cv2.imread(f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg')
    prev_img_bytes = cv2.imencode('.jpg', prev_img)[1].tobytes()

    svg_array = get_svg_data("all_printers/printer/00296.svg")
    svg_bytes = svg_array.tobytes()

    result = evaluate_layer.delay(img_bytes, prev_img_bytes, svg_bytes, svg_array.shape)

    result = result.get(propagate=False)
    # 'alerts': [{'value': 0.0, 'info': ''}, {'value': 1, 'info': '', 'error_type': 'WIPER_DEFECTED'}]
    img_bytes = result['visualizations'][0]
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('output.jpg', img)

    # SEND WORK TO back
    warns = []

    for alert in result['alerts']:
        warns.append({
            'error': REASON_TO_ID[alert['error_type']],
            'rate': alert['value']
        })

    layer_id = setup_layer(layer_number, '65a8c92d3b36820d63d99ff6', warns)
    add_photos_to_layer(layer_id, img_bytes, after_melting_img_bytes, svg_bytes)

    return 'image processed'
