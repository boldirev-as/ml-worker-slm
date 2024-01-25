import json

import requests
from tqdm import tqdm
from requests_for_backend import create_new_project


def create_request(layer_count_from_start, project_id, cur_layer_number):
    add_zeros = '0' * (5 - len(str(cur_layer_number)))
    add_zeros2 = '0' * (5 - len(str(cur_layer_number - 1)))

    svg_path = f"{project_name}/../svg/{add_zeros}{cur_layer_number}.svg"
    img_recoat_path = f'{project_name}/{add_zeros}{cur_layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{cur_layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{cur_layer_number - 1}-recoat.jpg'

    res = requests.get(URL,
                       params={
                           'project_id': project_id,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': layer_count_from_start
                       })
    if res.status_code >= 300:
        print(res.text)
        print(res, res.elapsed.total_seconds())


ALL_LAYERS = 1
URL = 'http://0.0.0.0:9080/start_processing/'
# PRINTER_DIR = '../../sirius/'
PRINTER_DIR = '../../all_printers/'

# ALL GOOD
# for layer_number in tqdm(range(200, 250)):
#     project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1

# LAZER
# for layer_number in tqdm(range(773, 790)):
#     project_name = PRINTER_DIR + '/print2/2023-12-06 18%3A42%3A35'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1

CONFIGURE = True
project_ids = {
    'METAL ABSENCE': '',
    'WIPER DEFECTED 1': '',
    'WIPER DEFECTED 2': ''
}

if CONFIGURE:
    # METAL ABSENCE
    project_id = create_new_project(3000, '123123', 'Деталь 1')
    for layer_number in tqdm(range(1067, 1967)):
        project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
        create_request(ALL_LAYERS, project_name, layer_number)
        ALL_LAYERS += 1
    project_ids['METAL ABSENCE'] = project_id

    # WIPER DEFECT reslice
    project_id = create_new_project(3000, '123123', 'Деталь 2')
    for layer_number in tqdm(range(300, 900)):
        project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
        create_request(ALL_LAYERS, project_name, layer_number)
        ALL_LAYERS += 1
    project_ids['WIPER DEFECTED 1'] = project_id

    # Wiper defect ignore
    project_id = create_new_project(3000, '123123', 'Деталь 3')
    for layer_number in tqdm(range(300, 900)):
        project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
        create_request(ALL_LAYERS, project_name, layer_number)
        ALL_LAYERS += 1
    project_ids['WIPER DEFECTED 2'] = project_id
else:
    with open('projects.json', mode='r') as f:
        project_ids = json.load(f)

    # METAL ABSENCE
    project_id = project_ids['METAL ABSENCE']
    project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
    create_request(900, project_name, 1967)

    # WIPER DEFECT reslice
    project_id = project_ids['WIPER DEFECTED 1']
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(600, project_name, 1049)

    # Wiper defect ignore
    project_id = project_ids['WIPER DEFECTED 2']
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(600, project_name, 1709)

with open('projects.json', mode='w') as f:
    json.dump(f, project_ids)

# ALL GOOD
# for layer_number in tqdm(range(700, 720)):
#     project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1
#
# # WIPER DEFECT reslice
# for layer_number in tqdm(range(1049, 1050)):
#     project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1
#
# # ALL GOOD
# for layer_number in tqdm(range(300, 350)):
#     project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1
#
# # Wiper defect ignore
# for layer_number in tqdm(range(1709, 1710)):
#     project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1
#
# # ALL GOOD
# for layer_number in tqdm(range(1700, 1710)):
#     project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
#     create_request(ALL_LAYERS, project_name, layer_number)
#     ALL_LAYERS += 1
