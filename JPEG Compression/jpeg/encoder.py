import PIL
import numpy as np
import pickle
import jpeg.utils as utils
import jpeg.huffman as huffman
from scipy.fftpack import dct

class Encoder:
    def __init__(self, image):
        self.image  = image
        self.width  = None
        self.height = None

    def dct(self, blocks):
        return np.array(dct(dct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho'))

    def quantization(self, matrix, type):
        if (type == 'l'):
            return(np.divide(matrix, utils.Q_Y).round().astype(np.int))
        elif (type == 'c'):
            return(np.divide(matrix, utils.Q_C).round().astype(np.int))
        return matrix

    def downsampling(self, matrix, k=2, type=2):
        """
        type=2 rows and columns reduction
        type=1 column reduction
        type=0 no reduction
        """
        new_image = matrix
        if type == 0:
            new_image = matrix
        elif type == 1:
            new_image = matrix[:,0::k]
        elif type == 2:
            new_image = matrix[0::k,0::k]

        return new_image

    def encoding(self):

        # Image width and height
        src_img_height, src_img_width = self.image.size
        print(f'Image: H = {src_img_height}, W = {src_img_width}')

        # Convert to numpy matrix
        src_img_mtx = np.asarray(self.image)

        # Convert 'RGB' to 'YCbCr'
        img_ycbcr = PIL.Image.fromarray(src_img_mtx).convert('YCbCr')
        img_ycbcr = np.asarray(img_ycbcr).astype(np.float64)

        # Convert to numpy array
        Y = img_ycbcr[:,:,0] - 128
        Cb = img_ycbcr[:,:,1] - 128
        Cr = img_ycbcr[:,:,2] - 128

        # Apply downsampling to Cb and Cr
        Cb = self.downsampling(Cb, src_img_height, src_img_width)
        Cr = self.downsampling(Cr, src_img_height, src_img_width)

        # Add zero-padding if needed
        Y = utils.add_zero_padding(Y)
        Cb = utils.add_zero_padding(Cb)
        Cr = utils.add_zero_padding(Cr)

        # Save new size
        self.height, self.width = Y.shape

        # Transform channels into blocks
        Y_bck = utils.transform_to_block(Y)
        Cb_bck = utils.transform_to_block(Cb)
        Cr_bck = utils.transform_to_block(Cr)

        # Calculate the DCT transform
        Y_dct = self.dct(Y_bck)
        Cb_dct = self.dct(Cb_bck)
        Cr_dct = self.dct(Cr_bck)

        # Quantization
        Y_q = self.quantization(Y_dct, 'l')
        Cb_q = self.quantization(Cb_dct, 'c')
        Cr_q = self.quantization(Cr_dct, 'c')
        
        # Creating huffman trees
        huffman_trees_for_Y = [huffman.HuffmanTree(np.array(block).flatten()) for block in Y_q]
        huffman_trees_for_Cb = [huffman.HuffmanTree(np.array(block).flatten()) for block in Cb_q]
        huffman_trees_for_Cr = [huffman.HuffmanTree(np.array(block).flatten()) for block in Cr_q]
        
        compressed_size_in_bits = 0
        actual_size_in_bits = 0

        for huffman_tree in huffman_trees_for_Y:
            result_size, actual_size = utils.compute_compression_rate(
                huffman_tree.value_to_bitstring_table(), huffman_tree.actual_arr
            )
            compressed_size_in_bits += result_size
            actual_size_in_bits += actual_size

        for huffman_tree in huffman_trees_for_Cb:
            result_size, actual_size = utils.compute_compression_rate(
                huffman_tree.value_to_bitstring_table(), huffman_tree.actual_arr
            )
            compressed_size_in_bits += result_size
            actual_size_in_bits += actual_size

        for huffman_tree in huffman_trees_for_Cr:
            result_size, actual_size = utils.compute_compression_rate(
                huffman_tree.value_to_bitstring_table(), huffman_tree.actual_arr
            )
            compressed_size_in_bits += result_size
            actual_size_in_bits += actual_size
        
        print('Actual Size: ', actual_size_in_bits)
        print('Compressed Size: ', compressed_size_in_bits)
        print('Compression Ratio: ', compressed_size_in_bits / actual_size_in_bits)
        
        # Storing huffman trees using pickle
        with open('huffman_trees', 'wb') as f:
            pickle.dump({
                'huffman_trees_for_Y': huffman_trees_for_Y,
                'huffman_trees_for_Cb': huffman_trees_for_Cb,
                'huffman_trees_for_Cr': huffman_trees_for_Cr
            }, f)