from queue import PriorityQueue

class HuffmanTree:

    class Node:
        def __init__(self, value, freq, left_child, right_child):
            self.value = value
            self.freq = freq
            self.left_child = left_child
            self.right_child = right_child

        @classmethod
        def init_leaf(self, value, freq):
            return self(value, freq, None, None)

        @classmethod
        def init_node(self, left_child, right_child):
            freq = left_child.freq + right_child.freq
            return self(None, freq, left_child, right_child)

        def is_leaf(self):
            return self.value is not None

        def __eq__(self, other):
            stup = self.value, self.freq, self.left_child, self.right_child
            otup = other.value, other.freq, other.left_child, other.right_child
            return stup == otup

        def __nq__(self, other):
            return not (self == other)

        def __lt__(self, other):
            return self.freq < other.freq

        def __le__(self, other):
            return self.freq < other.freq or self.freq == other.freq

        def __gt__(self, other):
            return not (self <= other)

        def __ge__(self, other):
            return not (self < other)

    def __init__(self, arr):
        self.actual_arr = arr
        q = PriorityQueue()

        for val, freq in self.calculate_freq(arr).items():
            q.put(self.Node.init_leaf(val, freq))

        while q.qsize() >= 2:
            u = q.get()
            v = q.get()

            q.put(self.Node.init_node(u, v))

        self.root = q.get()
        self.value_to_bitstring = dict()

    def calculate_freq(self, array):
        frequency = {key: 0 for key in array}
        for element in array:
            frequency[element] += 1
        return frequency

    def value_to_bitstring_table(self):
        if len(self.value_to_bitstring.keys()) == 0:
            self.create_huffman_table()
        return self.value_to_bitstring

    def create_huffman_table(self):
        def tree_traverse(current_node, bitstring=''):
            if current_node is None:
                return
            if current_node.is_leaf():
                self.value_to_bitstring[current_node.value] = bitstring
                return
            tree_traverse(current_node.left_child, bitstring + '0')
            tree_traverse(current_node.right_child, bitstring + '1')

        tree_traverse(self.root)
    
# import utils
# compressed_size_in_bits, actual_size_in_bits = utils.compute_compression_rate(HuffmanTree([10 ,34 ,10 ,8 ,10 ,10 ,127 ,43 ,6 ,34 ,10 ,5 ,34 ,8 ,8]).value_to_bitstring_table(), [10 ,34 ,10 ,8 ,10 ,10 ,127 ,43 ,6 ,34 ,10 ,5 ,34 ,8 ,8])
# print('Actual Size: ', actual_size_in_bits)
# print('Compressed Size: ', compressed_size_in_bits)
# print('Compression Ratio: ', compressed_size_in_bits / actual_size_in_bits)