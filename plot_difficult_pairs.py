

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


f = open('difficult_pairs.txt', 'r')
pairs = f.readlines()


# Load the images using matplotlib.image.imread
img_pairs = [[mpimg.imread(pair.split(',')[0]), mpimg.imread(pair.split(',')[1][:-1])] for pair in pairs]

plt.figure(figsize=(10000,10000))
# Create a figure with an lenght of pairs by 2 grid of subplots
fig, axs = plt.subplots(6, 2)

# Display each pair of images in a separate row of subplots
for i, (img1, img2) in enumerate(img_pairs):
    axs[i, 0].imshow(img1)
    axs[i, 0].axis('off')
    #axs[i, 0].set_title(f"Image {2*i+1}")
    axs[i, 1].imshow(img2)
    axs[i, 1].axis('off')
    
    #axs[i, 1].set_title(f"Image {2*i+2}")
    if i == 5:
        break

# Add a title to the figure

fig.suptitle("Comparison of Difficult Pairs")
fig.savefig("difficult_pairs.png")
