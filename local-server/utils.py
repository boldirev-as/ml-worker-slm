import io

import cairosvg
import cv2
from PIL import Image
from bs4 import BeautifulSoup
from scipy.special import erf
import numpy as np


def get_svg_data(svg_content: str):
    soup = BeautifulSoup(svg_content, 'xml')
    g_element = soup.find('g')
    if g_element:
        g_element.decompose()
    np_array_svg, bytes_svg = svg_to_numpy(str(soup))
    return np_array_svg.mean(axis=2) != 0, bytes_svg


def svg_to_numpy(svg_content):
    png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=1024, output_height=1024)
    image = Image.open(io.BytesIO(png_data))
    svg_numpy_array = np.rot90(np.fliplr(np.array(image)), k=3)
    return svg_numpy_array, png_data


def gaussian_stretching(pixel_value, mean, sigma, A):
    # return A * (1 / (1 + np.exp(-((pixel_value - mean) / (sigma / 3)))) - 0.5) * 2
    return A * (0.5 + 0.5 * erf((pixel_value - mean) / (sigma * np.sqrt(2))))


def processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = crop_image(img)

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


def crop_image(img):
    height, width = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    center = (height // 2, width // 2)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return img
