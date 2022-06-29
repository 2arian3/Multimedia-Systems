import sys
import jpeg
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: main.py <image_name>")
        sys.exit(1)
        
    # Reading Image
    image_name = sys.argv[1]
    img = jpeg.utils.read_image_from_file(image_name)
    
    encoder = jpeg.Encoder(img)
    encoder.encoding()

if __name__ == '__main__':
    main()