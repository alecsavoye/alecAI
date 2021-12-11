# worker functions that will be pulled into Main !
#

# some minor import overhead
#
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL

# because of lack of CPU optimization, reject errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# class for images that are processed, helps with management
#
class Img:

    # assign a name to the image
    def __init__(self, path, AS):
        self.path = path
        self.AS = AS
        # this will be what the actual image resides in
        self.carrier = tf.dtypes.DType
        print("Img object created with path {0} and is its "
              "classification for ASPHOTO is {1}".format(self.path, self.AS))

    def load_image(self, image_path, image_size=(2048, 1024)):
        # split an image into channels ...
        # NOTE ellipsis is a way of indexing thru multi-dim
        # ...numpy arrays!
        img = tf.io.decode_image(
            tf.io.read_file(image_path),
            channels=3, dtype=tf.float32)[tf.newaxis, ...]
        # rectify it for processing
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

        return img

    def visualize(self, images, titles=('',)):
        noi = len(images)
        image_sizes = [image.shape[1] for image in images]
        w = (image_sizes[0] * 6) // 320
        plt.figure(figsize=(w * noi, w))
        grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)

        for i in range(noi):
            plt.subplot(grid_look[i])
            plt.imshow(images[i][0], aspect='equal')
            plt.axis('off')
            plt.title(titles[i])
            plt.savefig('final.jpg')

        plt.show()

    # for actually exporting the result
    def export_image(self, tf_img):
        tf_img = tf_img * 255
        tf_img = np.array(tf_img, dtype=np.uint8)
        if np.ndim(tf_img)>3:
            assert tf_img.shape[0] == 1
            img = tf_img[0]

        return PIL.Image.fromarray(img)
