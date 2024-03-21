import os
from PIL import Image
import matplotlib.pyplot as plt
import math

def save_img(img,save_path):
    if img.ndim == 4:
        im = Image.fromarray(img[0])
    elif img.ndim == 3:
        im = Image.fromarray(img)
    else:
        raise ValueError("The dim of the picture is not right")
    im.save(save_path)
    return None

def save_images(img, nrow=1, ncol=None, save_path=None):
    batch_size, height, width, channel = img.shape
    if ncol == None:
        ncol = math.ceil(batch_size / nrow)
    for i in range(batch_size):
        im = Image.fromarray(img[i])
        file_name = os.path.join(save_path, str(i+1) + '.png')
        im.save(file_name)
    fig, axs = plt.subplots(nrow, ncol,figsize=(ncol, nrow))
    axs = axs.flatten()
    for i in range(batch_size):
        axs[i].imshow(img[i])
    for i in range(ncol*nrow):
        axs[i].axis('off')
    fig.savefig(os.path.join(save_path, 'total.png'),dpi=300)
    return None