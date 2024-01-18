import io

import cairosvg
import cv2
from PIL import Image
from bs4 import BeautifulSoup
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt


def get_svg_data(svg_path):
    with open(svg_path) as svg_file:
        svg_content = svg_file.read()
    soup = BeautifulSoup(svg_content, 'xml')
    g_element = soup.find('g')
    if g_element:
        g_element.decompose()
    return svg_to_numpy(str(soup)).mean(axis=2) != 0


def svg_to_numpy(svg_content):
    png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=1024, output_height=1024)
    image = Image.open(io.BytesIO(png_data))
    svg_numpy_array = np.rot90(np.fliplr(np.array(image)), k=3)
    return svg_numpy_array


def get_error_image(svg_data, defects_data, intersection_data, show=False):
    error_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    error_image[svg_data != 0] = [0, 255, 0]  # RGB -> BGR
    error_image[defects_data != 0] = [0, 255, 255]
    error_image[intersection_data != 0] = [0, 0, 255]
    if show: plt.imshow(error_image)
    return error_image


def get_overlapped_image(image_data, error_image):
    base_image = Image.fromarray(image_data).resize((1024, 1024)).convert("RGBA")
    overlay_image = Image.fromarray(error_image).convert("RGBA")
    mask = Image.fromarray((error_image.sum(axis=2) != 0).astype('uint8') * 60)
    overlay_image.putalpha(mask)
    base_image.paste(overlay_image, mask=mask)
    return base_image


def get_defects_info(results, svg_data, image_data, show=False):
    if results[0].masks is None:
        return 0, Image.fromarray(image_data)

    defects_data = (results[0].masks.cpu().numpy().data.transpose(1, 2, 0).sum(axis=2) != 0).astype(np.uint8)
    intersection_data = np.bitwise_and(svg_data, defects_data)
    error_ratio = np.count_nonzero(intersection_data) / np.count_nonzero(svg_data)
    error_image = get_error_image(svg_data, defects_data, intersection_data, show=show)
    overlapped_image = get_overlapped_image(image_data, error_image)
    return error_ratio, overlapped_image


def gaussian_stretching(pixel_value, mean, sigma, A):
    # return A * (1 / (1 + np.exp(-((pixel_value - mean) / (sigma / 3)))) - 0.5) * 2
    return A * (0.5 + 0.5 * erf((pixel_value - mean) / (sigma * np.sqrt(2))))


def processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f1 = 6
    height, width = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    center = (height // 2, width // 2)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    mean = np.mean(img)
    sigma = np.std(img)
    A = 255  # Коэффициент масштабирования
    # Применение гауссовского растяжения гистограммы
    out = gaussian_stretching(img, mean, sigma, A)
    # Приведение значений к диапазону [0, 255]
    out = np.clip(out, 0, 255).astype(np.uint8)
    img = cv2.bilateralFilter(out, 5, 120, 120)
    backtorgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return backtorgb
