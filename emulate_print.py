import requests

for layer_num in range(1500, 2000):
    res = requests.get('http://0.0.0.0:6666/start_processing/',
                       params={
                           'project_name': 'all_printers/print4/2024-01-04 15%3A28%3A26/',
                           'layer_number': layer_num})
    print(res, res.text, res.elapsed.total_seconds())
