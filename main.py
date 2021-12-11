# ALEC SAVOYE
# AAN FOR ALECSAVOYE.COM / PHOTOGRAPHY PROJECT

# Import some needed resources ...
#

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL

from worker import Img


# Process Incoming Images
#

# instantiate two Img objects for inputs
original_image = Img("imgin/original_image.jpg", False)
style_image = Img("imgin/style_image.jpg", True)
res_nm = str( input ("input desired result name (+ return): "))

# fill out their dtype carriers !
original_image.carrier = original_image.load_image(original_image.path)
style_image.carrier = style_image.load_image(style_image.path)

# training the neural network ...
# using average pool network - see paper printed 12/7/21

style_image.carrier = tf.nn.avg_pool(
    style_image.carrier,
    ksize = [3,3],
    strides = [1,1],
    padding = 'VALID'
)

# visualization function to show results

original_image.visualize(

    [original_image.carrier, style_image.carrier],
    ['Original Image', 'Style Image']

)

# commence arbitrary stylization of the model !

stylize_model = tf_hub.load('tf_model')

results = stylize_model(
    tf.constant(original_image.carrier),
    tf.constant(style_image.carrier)
)

stylized_photo = results[0]

original_image.visualize(
    # export content!
    [original_image.carrier, style_image.carrier, stylized_photo],
    titles = ['Original', 'Style', 'Stylized']
)

#final export to project directory
original_image.export_image(stylized_photo).save("imgout/" + res_nm + ".png")

