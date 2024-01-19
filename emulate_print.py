import requests
from tqdm import tqdm

for layer_num in tqdm(range(2000, 2500)):
    res = requests.get('http://0.0.0.0:8090/start_processing/',
                       params={
                           'project_name': 'all_printers/print4/2024-01-04 15%3A28%3A26/',
                           'layer_number': layer_num})
    # if res.status_code >= 300:
    print(res, res.text, res.elapsed.total_seconds())
