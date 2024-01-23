import httpx
from httpx import Client

URL_BACKEND = 'http://158.160.126.165:8000/'


def create_new_printer(uid='213213', name='SLM 3d Midi', description='my little pony'):
    with Client() as client:
        response = client.post(URL_BACKEND + 'printers',
                               json={
                                   'uid': uid,
                                   'name': name,
                                   'description': description
                               })
    return response


def create_new_project(layers_len=100, printer_uid='213213',
                       name='Project for creating a little pony'):
    with Client() as client:
        response = client.post(URL_BACKEND + 'projects/',
                               json={
                                   'printer_uid': printer_uid,
                                   'name': name,
                                   'layers_len': layers_len
                               })
    return response.json()['id']


def setup_layer(layer_num, project_id, warns, recommendation="fix"):
    with Client() as client:
        response = client.post(URL_BACKEND + 'create_layer',
                               json={
                                   "order": layer_num,
                                   "project_id": project_id,
                                   "warns": warns,
                                   "recommendation": recommendation
                               })
    if response.status_code >= 300:
        print(response.text)
    return response.json()['id']


def add_photos_to_layer(layer_id, before_melting_img, after_melting_img, svg_img):
    with Client() as client:
        response = client.post(URL_BACKEND + 'add_photos_to_layer',
                               data={
                                   'layer_id': layer_id,
                               },
                               files=[
                                   ('before_melting_image', ('before.jpg', before_melting_img)),
                                   ('after_melting_image', ('after.jpg', after_melting_img)),
                                   ('svg_image', ('svg.png', svg_img)),
                               ])
    return response


if __name__ == '__main__':
    # setup_layer(0, '65a8c92d3b36820d63d99ff6', [])
    # 65a8c92d3b36820d63d99ff6
    create_new_printer()
    project_id = create_new_project(3000)
    # project_id = '65a93274fa30c179992ab501'
    #
    print(project_id)
    #
    # layer_id = setup_layer(0, project_id, [], main_client)
    # print(layer_id)
    #
    # img = cv2.imread('all_printers/printer/print4_01251-recoat.jpeg')
    # img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
    #
    # prev_img = cv2.imread('all_printers/printer/00442-recoat.jpg')
    # prev_img_bytes = cv2.imencode('.jpg', prev_img)[1].tobytes()
    #
    # # svg_array = get_svg_data("printer/00296.svg")
    # # rgb_svg_bytes = cairosvg.svg2png(bytestring=svg_array.encode('utf-8'), output_width=1024, output_height=1024)
    # # svg_bytes = svg_array.tobytes()
    #
    # svg_bytes = open("all_printers/printer/00296.svg", mode='rb')
    # prev_img.tobytes()
    #
    # print(type(layer_id), type(img_bytes), type(prev_img_bytes), type(svg_bytes))
    #
    # add_photos_to_layer(layer_id, img_bytes, prev_img_bytes, svg_bytes)
