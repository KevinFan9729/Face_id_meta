import matplotlib.image as mpimg
import matplotlib.pyplot as plt

f = open('difficult_pairs_ssim.txt', 'r')
pairs = f.readlines()

# Load the images using matplotlib.image.imread
img_pairs = [[pair.split(',')[0], pair.split(',')[1][:-1]] for pair in pairs]

# Create a figure with an lenght of pairs by 2 grid of subplots
fig, axs = plt.subplots(6, 2)

c = 0
# Display each pair of images in a separate row of subplots
for i in [0, 1, 1000, 1001, 4000, 4001]:

    #for i, (img1, img2) in enumerate(img_pairs):
    print(img_pairs[i][0], img_pairs[i][1])
    img1, img2 = mpimg.imread(img_pairs[i][0]), mpimg.imread(img_pairs[i][1])
    axs[c, 0].imshow(img1)
    axs[c, 0].axis('off')
    #axs[i, 0].set_title(f"Image {2*i+1}")
    axs[c, 1].imshow(img2)
    axs[c, 1].axis('off')
    c += 1
    #axs[i, 1].set_title(f"Image {2*i+2}")

# Add a title to the figure

fig.suptitle("Comparison of Difficult Pairs")
fig.savefig("difficult_pairs_ssim.png")
