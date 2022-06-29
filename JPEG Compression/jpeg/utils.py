from PIL import Image
import numpy as np

# Quantization matrix for luminance
Q_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

# Quantization matrix for chrominance
Q_C = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])


def read_image_from_file(filename):
    return Image.open(filename)

def add_zero_padding(matrix):
    n_col, n_row = matrix.shape[0], matrix.shape[1]

    if (n_col % 8 != 0):
        img_width = n_col // 8 * 8 + 8
    else:
        img_width = n_col

    if (n_row % 8 != 0):
        img_height = n_row // 8 * 8 + 8
    else:
        img_height = n_row

    new_matrix = np.zeros((img_width, img_height), dtype=np.float64)
    for y in range(n_col):
        for x in range(n_row):
            new_matrix[y][x] = matrix[y][x]

    return new_matrix

def transform_to_block(image):
    img_w, img_h = image.shape
    blocks = []
    for i in range(0, img_w, 8):
        for j in range(0, img_h, 8):
            blocks.append(image[i:i+8,j:j+8])

    return blocks

def compute_compression_rate(huffman_codes, data, data_default_size=8):
    result = sum([len(huffman_codes[datum]) for datum in data])
    actual_size = data_default_size * len(data)
    return result, actual_size