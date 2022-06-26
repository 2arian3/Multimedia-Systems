import PIL
import numpy as np
import matplotlib.pyplot as plt
import sys

 
def read_image(dir):
    img = PIL.Image.open(dir)
    return np.asarray(img)


def save_image(image, name):
    img = PIL.Image.fromarray(image)
    img.save(name)
    

def convert_rgb_to_grayscale(image):
    return np.uint8(np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]))


def image_histogram(image):
    n, m = image.shape
    hist = np.zeros(256)
    for i in range(n):
        for j in range(m):
            hist[image[i, j]] += 1
    return hist


def cumulative_sum(array):
    return np.array([sum(array[:i+1]) for i in range(len(array))])


def histogram_equalization(image):
    img = convert_rgb_to_grayscale(image)
    n, m = img.shape
    
    hist = image_histogram(img)
    cumulative_sum_of_colors = cumulative_sum(hist)
    transform_function = np.uint8((255 * cumulative_sum_of_colors) / (n * m))
    
    new_image = np.zeros_like(img)
    
    for i in range(n):
        for j in range(m):
            new_image[i, j] = transform_function[img[i, j]]
        
    return new_image, hist, cumulative_sum_of_colors


def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram_equalizer.py <image_path>")
        sys.exit(1)
    
    image = read_image(sys.argv[1])
    new_image, hist, cumulative_sum_of_colors = histogram_equalization(image)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(new_image, cmap='gray')
    plt.title('Histogram equalization')
    plt.show()
    
    save_image(new_image, 'histogram_equalized.png')
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(hist)), hist, color='red')
    plt.title('Original Image Histogram')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(cumulative_sum_of_colors)), cumulative_sum_of_colors, color='blue')
    plt.title('Original Image Cumulative Histogram')
    plt.show()
    
    new_image_histogram = image_histogram(new_image)
    new_image_cumulative_sum_of_colors = cumulative_sum(new_image_histogram)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(new_image_histogram)), new_image_histogram, color='red')
    plt.title('Result Image Histogram')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(new_image_cumulative_sum_of_colors)), new_image_cumulative_sum_of_colors, color='blue')
    plt.title('Result Image Cumulative Histogram')
    plt.show()
    

if __name__ == '__main__':
    main()
