import requests
from tqdm import tqdm

ALL_LAYERS = 1

for layer_number in tqdm(range(100, 150)):

    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    project_name = '../../sirius/print4/2024-01-04 15%3A28%3A26'
    svg_path = f"{project_name}/../svg/{add_zeros}{layer_number}.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    res = requests.get('http://0.0.0.0:9080/start_processing/',
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': ALL_LAYERS})
    ALL_LAYERS += 1
    if res.status_code >= 300:
        print(res, res.text, res.elapsed.total_seconds())

for layer_number in tqdm(range(2017, 2020)):
    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    project_name = '../../sirius/print3_fail_with_1967/2023-12-15 18%3A51%3A07'
    svg_path = f"{project_name}/../svg/{add_zeros}{layer_number}.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    res = requests.get('http://0.0.0.0:9080/start_processing/',
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': ALL_LAYERS})
    ALL_LAYERS += 1
    if res.status_code >= 300:
        print(res, res.text, res.elapsed.total_seconds())

for layer_number in tqdm(range(1000, 1050)):

    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    project_name = '../../sirius/print4/2024-01-04 15%3A28%3A26'
    svg_path = f"{project_name}/../svg/{add_zeros}{layer_number}.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    res = requests.get('http://0.0.0.0:9080/start_processing/',
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': ALL_LAYERS})
    ALL_LAYERS += 1
    if res.status_code >= 300:
        print(res, res.text, res.elapsed.total_seconds())

for layer_number in tqdm(range(1708, 1710)):

    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    project_name = '../../sirius/print4/2024-01-04 15%3A28%3A26'
    svg_path = f"{project_name}/../svg/{add_zeros}{layer_number}.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    res = requests.get('http://0.0.0.0:9080/start_processing/',
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': ALL_LAYERS})
    ALL_LAYERS += 1
    if res.status_code >= 300:
        print(res, res.text, res.elapsed.total_seconds())

for layer_number in tqdm(range(1000, 1050)):
    add_zeros = '0' * (5 - len(str(layer_number)))
    add_zeros2 = '0' * (5 - len(str(layer_number - 1)))

    project_name = '../../sirius/print3_fail_with_1967/2023-12-15 18%3A51%3A07'
    svg_path = f"{project_name}/../svg/{add_zeros}{layer_number}.svg"  # TODO: create normal svg filename
    img_recoat_path = f'{project_name}/{add_zeros}{layer_number}-recoat.jpg'
    img_scan_path = f'{project_name}/{add_zeros}{layer_number}-scan.jpg'
    img_recoat_prev_path = f'{project_name}/{add_zeros2}{layer_number - 1}-recoat.jpg'

    res = requests.get('http://0.0.0.0:9080/start_processing/',
                       params={
                           'project_name': project_name,
                           'svg_path': svg_path,
                           'img_recoat_path': img_recoat_path,
                           'img_scan_path': img_scan_path,
                           'img_recoat_prev_path': img_recoat_prev_path,
                           'layer_number': ALL_LAYERS})
    ALL_LAYERS += 1
    if res.status_code >= 300:
        print(res, res.text, res.elapsed.total_seconds())
