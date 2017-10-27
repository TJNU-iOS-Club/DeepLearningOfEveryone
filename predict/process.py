import numpy as np
from PIL import Image

image_sequence = ['a', 'b', 'c', 'd', 'e']
value_sequence = []

for code in image_sequence:
    im = Image.open('./%s.jpg' % code)
    im.thumbnail((28, 28))
    value = np.array(im).reshape(28, 28, 1)
    value = value / 255
    value_sequence.append(value)

value_array = np.array(value_sequence)
np.save('./session-10-27', value_array)
