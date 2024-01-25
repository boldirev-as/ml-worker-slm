import json

import requests
from httpx import Client
from tqdm import tqdm


def new_project(layers_len=100,
                name='Project for creating a little pony'):
    with Client() as client:
        response = client.get(URL + 'create_project',
                              params={
                                  'project_name': name,
                                  'total_number_layers': layers_len
                              })
        if response.status_code >= 300:
            print(response.json(), response.text)
    return response.json()['project_id']


def create_request(layer_count_from_start, project_id, cur_layer_number, drop_last_state=False):
    add_zeros = '0' * (5 - len(str(cur_layer_number)))
    add_zeros2 = '0' * (5 - len(str(cur_layer_number - 1)))

    svg_path = f"{project_name}/../svg/{add_zeros}{cur_layer_number}.svg"
    img_recoat_path = f'{project_name}/{add_zeros}{cur_layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{cur_layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{cur_layer_number - 1}-recoat.jpg'

    res = requests.get(URL + 'start_processing',
                       params={
                           'project_id': project_id,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': layer_count_from_start,
                           'drop_last_state': drop_last_state
                       })
    if res.status_code >= 300:
        print(res.text)
        print(res, res.elapsed.total_seconds())


ALL_LAYERS = 1
URL = 'http://0.0.0.0:9080/'
# PRINTER_DIR = '../../sirius/'
PRINTER_DIR = '../all_printers/'

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

CONFIGURE = False
project_ids = {
    'METAL ABSENCE': '',
    'WIPER DEFECTED 1': '',
    'WIPER DEFECTED 2': '',
    'LENGTH': 2
}

if CONFIGURE:
    # METAL ABSENCE
    project_id = new_project(3000, 'Деталь 1')
    ALL_LAYERS = 1
    for layer_number in tqdm(range(1967 - project_ids['LENGTH'], 1967)):
        project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
        create_request(ALL_LAYERS, project_id, layer_number)
        ALL_LAYERS += 1
    project_ids['METAL ABSENCE'] = project_id

    # WIPER DEFECT reslice
    project_id = new_project(3000, 'Деталь 2')
    ALL_LAYERS = 1
    for layer_number in tqdm(range(800 - project_ids['LENGTH'], 800)):
        project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
        create_request(ALL_LAYERS, project_id, layer_number)
        ALL_LAYERS += 1
    project_ids['WIPER DEFECTED 1'] = project_id

    # Wiper defect ignore
    project_id = new_project(3000, 'Деталь 3')
    ALL_LAYERS = 1
    for layer_number in tqdm(range(700 - project_ids['LENGTH'], 700)):
        project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
        create_request(ALL_LAYERS, project_id, layer_number)
        ALL_LAYERS += 1
    project_ids['WIPER DEFECTED 2'] = project_id
else:
    with open('projects.json', mode='r') as f:
        project_ids = json.load(f)

    # METAL ABSENCE
    project_id = project_ids['METAL ABSENCE']
    project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
    create_request(project_ids['LENGTH'] + 1, project_id, 1967, True)

    # WIPER DEFECT reslice
    project_id = project_ids['WIPER DEFECTED 1']
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(project_ids['LENGTH'] + 1, project_id, 1101, True)

    # Wiper defect ignore
    project_id = project_ids['WIPER DEFECTED 2']
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(project_ids['LENGTH'] + 1, project_id, 1709, True)

with open('projects.json', mode='w') as f:
    json.dump(project_ids, f)

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
