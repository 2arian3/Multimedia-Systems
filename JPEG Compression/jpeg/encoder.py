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
        C_q = np.vstack((Cb_q, Cr_q))
        
        Y_DPCM, Y_RLC = [], []
        C_DPCM, C_RLC = [], []
        
        for i in utils.ZIGZAG:
            y_block = np.array(Y_q[i]).flatten()
            c_block = np.array(C_q[i]).flatten()
            
            Y_DPCM.append(y_block[0])
            C_DPCM.append(c_block[0])

            Y_RLC.append(run_length_coding(y_block[1:]))
            C_RLC.append(run_length_coding(c_block[1:]))
        
        Y_DPCM = encode_differential(Y_DPCM)
        C_DPCM = encode_differential(C_DPCM)
        
        result = []
        for rlc in Y_RLC:
            result += rlc
        Y_RLC = result
        
        result = []
        for rlc in C_RLC:
            result += rlc
        C_RLC = result
        
        Y_DPCM = [(len(get_binary_string(data)), get_binary_string(data)) for data in Y_DPCM]
        C_DPCM = [(len(get_binary_string(data)), get_binary_string(data)) for data in C_DPCM]
   
        # Creating huffman trees
        huffman_tree_for_y_dc = huffman.HuffmanTree([size for size, _ in Y_DPCM])
        huffman_tree_for_c_dc = huffman.HuffmanTree([size for size, _ in C_DPCM])
        
        huffman_tree_for_y_ac = huffman.HuffmanTree([run_length for run_length, _ in Y_RLC])
        huffman_tree_for_c_ac = huffman.HuffmanTree([run_length for run_length, _ in C_RLC])
        
        # Storing huffman trees using pickle
        with open('huffman_trees', 'wb') as f:
            pickle.dump({
                'huffman_tree_for_y_dc': huffman_tree_for_y_dc,
                'huffman_tree_for_y_ac': huffman_tree_for_y_ac,
                'huffman_tree_for_c_dc': huffman_tree_for_c_dc,
                'huffman_tree_for_c_ac': huffman_tree_for_c_ac
            }, f)
            

def run_length_coding(data):
    """
    Run length encoding
    """
    result = []
    zero_counts = 0
    for i in range(len(data)):
        if data[i] == 0:
            zero_counts += 1
        else:
            result.append((zero_counts, data[i]))
            zero_counts = 0
    result.append((0, 0))
    return result

def encode_differential(seq):
    return list(
        (item - seq[i - 1]) if i else item
        for i, item in enumerate(seq)
    )
    
def get_binary_string(num):
    string = bin(num)[2:]
    if num < 0:
        return ''.join(['1' if char == '0' else '0' for char in string])
    return string