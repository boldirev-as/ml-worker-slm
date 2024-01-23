import requests
from tqdm import tqdm


def create_request(layer_count, project_name):
    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    svg_path = f"{project_name}/../svg/{add_zeros}{layer_number}.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    res = requests.get(URL,
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': layer_count})
    if res.status_code >= 300:
        print(res.text)
    print(res, res.elapsed.total_seconds())


ALL_LAYERS = 1
URL = 'http://0.0.0.0:9080/start_processing/'
# PRINTER_DIR = '../../sirius/'
PRINTER_DIR = '../all_printers/'

for layer_number in tqdm(range(147, 150)):
    project_name = PRINTER_DIR + '/print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name)
    ALL_LAYERS += 1

# for layer_number in tqdm(range(2017, 2020)):
#     project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
#     create_request(ALL_LAYERS, project_name)
#     ALL_LAYERS += 1

for layer_number in tqdm(range(1047, 1050)):
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name)
    ALL_LAYERS += 1

for layer_number in tqdm(range(1708, 1710)):
    project_name = PRINTER_DIR + 'print4/2024-01-04 15%3A28%3A26'
    create_request(ALL_LAYERS, project_name)
    ALL_LAYERS += 1

# for layer_number in tqdm(range(1708, 1710)):
#     project_name = PRINTER_DIR + 'print3_fail_with_1967/2023-12-15 18%3A51%3A07'
#     create_request(ALL_LAYERS, project_name)
#     ALL_LAYERS += 1
