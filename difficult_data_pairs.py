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
    output_file.write(pair[0], pair[1])

output_file.close()

# Load the images using matplotlib.image.imread
img_pairs = [[mpimg.imread(pair[0]), mpimg.imread(pair[1])] for pair in pairs]

# Create a figure with an lenght of pairs by 2 grid of subplots
fig, axs = plt.subplots(len(pairs), 2)

# Display each pair of images in a separate row of subplots
for i, (img1, img2) in enumerate(img_pairs):
    axs[i, 0].imshow(img1)
    axs[i, 0].set_title(f"Image {2*i+1}")
    axs[i, 1].imshow(img2)
    axs[i, 1].set_title(f"Image {2*i+2}")

# Add a title to the figure
fig.suptitle("Comparison of Difficult Pairs")
fig.savefig("difficult_pairs.png")

