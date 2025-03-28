import numpy as np

def pos_encoding(position: int, d_model: int):

    # positional encoding (pos, 2*i) = sin(pos/((10000)^(2*i/d_model)))
    # positional encoding (pos, 2*i+1) = cos(pos/((10000)^(2*i/d_model)))

    i = np.arange(0, d_model, 2)
    exponent = 10000**((i)/d_model) # dont need to do 2i since in the above .arrange, it is in steps of 2
    pos_encoding = np.zeros((position, d_model))  # Store all positions

    for pos in range(position):
        pos_encoding[pos, 0::2] = np.sin(pos / exponent)  # Even indices
        pos_encoding[pos, 1::2] = np.cos(pos / exponent)  # Odd indices
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding
