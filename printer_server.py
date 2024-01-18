import cv2
from fastapi import FastAPI
import configparser

import numpy as np
from httpx import Client
from starlette.responses import JSONResponse

from tasks import evaluate_layer
from requests_for_backend import setup_layer, add_photos_to_layer, create_new_project
from utils import get_svg_data

config = configparser.ConfigParser()
config.read("settings.ini")

main_client = Client()
CURRENT_PROJECT_DETAILS = {'ID': '65a93274fa30c179992ab501'}

REASON_TO_ID = {
    'WIPER_DEFECTED': 0,
    'METAL_ABSENCE': 1
}

app = FastAPI()


@app.get("/start_processing/")
def start(project_name: str, layer_number: int):
    if layer_number == 1:
        CURRENT_PROJECT_DETAILS['ID'] = create_new_project(main_client, 3000)

    if layer_number < 40:
        return {"warns": [], "project_id": CURRENT_PROJECT_DETAILS['ID'], "order": layer_number,
                "printer_uid": "", "before_melting_image": "",
                "after_melting_image": "",
                "svg_image": "", "id": ""}

    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    # PATH PATH PATH
    svg_path = "all_printers/printer/00296.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    img = cv2.imread(img_recoat_path)
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()

    after_melting_img = cv2.imread(img_scan_path)
    after_melting_img_bytes = cv2.imencode('.jpg', after_melting_img)[1].tobytes()

    prev_img = cv2.imread(img_recoat_prev_path)
    prev_img_bytes = cv2.imencode('.jpg', prev_img)[1].tobytes()

    svg_array = get_svg_data(svg_path)
    svg_bytes = svg_array.tobytes()

    result = evaluate_layer.delay(img_bytes, prev_img_bytes, svg_bytes, svg_array.shape)

    try:
        result = result.get()
    except Exception as e:
        return JSONResponse(
            status_code=418,
            content={"message": f"Oops! Celery did something: {e}"},
        )

    # 'alerts': [{'value': 0.0, 'info': ''}, {'value': 1, 'info': '', 'error_type': 'WIPER_DEFECTED'}]
    img_bytes = result['visualizations'][0]
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('output.jpg', img)

    # SEND WORK TO back
    warns = []

    for alert in result['alerts']:
        warns.append({
            'reason': REASON_TO_ID[alert['error_type']],
            'rate': alert['value']
        })

    print(warns)

    svg_bytes = open(svg_path, mode='rb').read()

    layer_id = setup_layer(layer_number, CURRENT_PROJECT_DETAILS['ID'], warns, main_client)
    response = add_photos_to_layer(layer_id, img_bytes, after_melting_img_bytes, svg_bytes, main_client)

    print('image processed', response.text)

    return response.json()
