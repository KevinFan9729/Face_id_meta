import os
from datetime import datetime

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
for pair in pairs:
    print(pair[0].split('\\')[-2:], pair[1].split('\\')[-2:])
    print(pair)
    print()