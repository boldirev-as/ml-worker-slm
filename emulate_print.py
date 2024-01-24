import requests
from tqdm import tqdm


def create_request(layer_count_from_start, project_name, cur_layer_number):
    add_zeros = '0' * (5 - len(str(cur_layer_number)))
    add_zeros2 = '0' * (5 - len(str(cur_layer_number - 1)))

    svg_path = f"{project_name}/../svg/{add_zeros}{cur_layer_number}.svg"
    img_recoat_path = f'{project_name}/{add_zeros}{cur_layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{cur_layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{cur_layer_number - 1}-recoat.jpg'

    res = requests.get(URL,
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': layer_count_from_start})
    if res.status_code >= 300:
        print(res.text)
        print(res, res.elapsed.total_seconds())


ALL_LAYERS = 1
URL = 'http://0.0.0.0:9080/start_processing/'
# PRINTER_DIR = '../../sirius/'
PRINTER_DIR = '../all_printers/'

# ALL GOOD
for layer_number in tqdm(range(200, 250)):
    project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1

# METAL ABSENCE
for layer_number in tqdm(range(1967, 1968)):
    project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1

# ALL GOOD
for layer_number in tqdm(range(700, 720)):
    project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1

# WIPER DEFECT reslice
for layer_number in tqdm(range(1049, 1050)):
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1

# ALL GOOD
for layer_number in tqdm(range(300, 350)):
    project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1

# Wiper defect ignore
for layer_number in tqdm(range(1709, 1710)):
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1

# ALL GOOD
for layer_number in tqdm(range(1700, 1710)):
    project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
    create_request(ALL_LAYERS, project_name, layer_number)
    ALL_LAYERS += 1
