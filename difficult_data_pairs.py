import os
import modules.util as util

home=os.path.abspath(os.getcwd())
data_path=os.path.join(home, 'data_shortlist')

#global data
pairs=[]
classes=[]

for file in os.listdir(data_path):
    classes.append(file)

util.make_pairs(data_path, pairs, classes)
print(pairs)
print(classes)