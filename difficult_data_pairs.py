import os
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import modules.util as util

home = os.path.abspath(os.getcwd())
data_path = os.path.join(home, 'data_train_valset')  #data_shortlist

#global data
pairs = []
classes = []

for file in os.listdir(data_path):
    classes.append(file)
print("number of classes to process: ", len(classes))

t1 = datetime.now()
util.make_pairs(data_path, pairs, classes)
t2 = datetime.now()
print("time difference: ", t2 - t1)

output_file = open('difficult_pairs.txt', 'w')
for pair in pairs:
    output_file.write(pair[0] + ',' + pair[1] + '\n')

output_file.close()
