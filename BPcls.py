# classes for backpropogation !
# update 12/9 - classes decided not to be used because
# they would not enable easy indexing when summing up the contributions
# from a given layer. z^L = sigma (w^L-1) for all in layer
# considered doing a linked list but that would introduce a lot of complexity
# going to try using 2D / 3D array, and implement classes as found practical
# ... for recursion, the latter might be quite a bit better

# getting module for making neurons initial weights random
import random

class Network:

    # properties that will make it easier to track props
    # similar operation not performed for neurons because
    # of concerns for indexing thru a class object
    def __init__(self, hl, lsz, isz, osz):
        self.contents = []
        self.hl = hl
        self.lsz = lsz
        self.isz = isz
        self.osz = osz

# IMPORTANT NOTE ON NEURONS:
# the weight attribute of a neuron object represents
# the weights of prev layers neurons tied to that
# given neuron. with varying layer sizes this will change.
# (IE output neuron must have hidden layer number of neurons for
# number of weights, or first hidden layer must have input layer
# number of neurons for number of weights.

class Neuron:

    def __init__(self, wtnum, lyr):
        # layer information will be useful for fp and bp...
        self.lyr = lyr

        # initialize activation as a float
        self.act = 0.0
        # weights will be a list, corresp to next network

        self.weights = []
        i = 0
        while i < wtnum:
            self.weights.append((round(random.random(), 3)))
            i += 1

        # bias, which will be used to calculate activation
        self.bias = 0