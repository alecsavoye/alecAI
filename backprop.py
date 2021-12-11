# BACKPROPOGATION PRACTICE
# this version to have X layers total
# INPUT
# (however many hidden layers...)
# output
# in each layer, X neurons

# importing needed modules
#
# for generating random numbers while initializing net
import random

# for network property tracking through various levels
import BPcls as bpc

# for sigmoid function exponential term & summing intermediate vals
from math import exp
from math import fsum

# create the network, completely empty ...
# our neural network will be a 2D array of arrays...
# wherein the indices of each array will be their own array,
# representing a neuron. first entry: activation, second entry, weight
# HL is number of hidden layers!

def createnet(hl, lsz, isz, osz):

    # create that network !
    # NEED TO FIX NEURONS TO HAVE WEIGHTS FOR ALL NEXT LAYER NEURONS!

    network = bpc.Network(hl, lsz, isz, osz)

    # input layer will be as defined...
    # by definition, the input neurons do not have weights associated wiht them
    # (since weights in this scheme are stored in the receiving end neuron)
    # INPUT LAYER IS LAYER # 0
    network.contents.append(createlay(isz, 0, 0))

    i1 = 1
    # first hidden layer will corresp to # neurons in input layer
    while i1 < 2:

        network.contents.append(createlay(lsz, isz, i1))
        i1 += 1

    # the rest of the hidden layers will corresp to same
    # number of neurons as the previous layer
    while i1 < hl + 1:

        network.contents.append(createlay(lsz, lsz, i1))
        i1 += 1

    # output layer is the last step...
    # the output layer neurons will have the same number
    # of weights as the final hidden layer has neurons
    network.contents.append(createlay(osz, lsz, hl+1))

    return network

# facilitates the creation of neurons within other functions
# rendered obsolete when classes were implemented for neurons
"""
def createneu(lsz):
    i = 0
    neuron = [0]
    while i < lsz:
        neuron.append(round(random.random(), 3))
        i += 1

    return neuron
"""

# lsz is how many neurons in a layer
# nwts is how many weights a given neuron will have to have
# derived from the number of neurons in the previous layer
def createlay(lsz,wtnum,lnum):

    i = 0
    lyr = []
    # if we're in input layer, don't waste resources
    # assigning weights that will never be used
    if lnum == 0:
        while i < lsz:
            lyr.append(bpc.Neuron(0, 0))
            i += 1
    else:
        while i < lsz:
###           print("line 87, lnum = {0}".format(lnum))
            lyr.append(bpc.Neuron(wtnum, lnum))
            i += 1

    return lyr

# forward propogation is the first step where a network output is obtained
# then, cost function can be calculated and backpropogation can take place
# start at output neuron layer, and work backwards
# ... logically you need the activation of the previous but then activations before that, etc.

def fprop(network, tr):
    # feed in the inputs to activations of input layer neurons ...
    for i in range(0, len(network.contents[0])):
        network.contents[0][i].act = tr[0][i]
    for neuron in network.contents[network.hl + 1]:
        fp_help(neuron, network)


# this recursive helper function will calculate the activation
# of the output neurons, ultimately, but this requires figuring
# out the activation of the previous! etc up to the input layer
# ... calculates temporary sum !

def fp_help(neuron, network):
    if neuron.lyr > 1:
        # bumble all the way up to the first layer after input,
        # and update the activations there!
        # then the program will proceed down the line updating
            # activations until it lands back at output

        for prevneu in network.contents[neuron.lyr - 1]:
            fp_help(prevneu, network)

    else:
        # mutliply all weights of ties to other neurons times acts
        # of other neurons...

        # weights for this neuron, which should hopefully
        # match in index to previous layer's neurons
        temp1 = neuron.weights
        # activations of preivous layer's neurons
        temp2 = [i.act for i in network.contents[neuron.lyr - 1]]
        # temp sum before sigmoid
        smlist = [a * b for a,b in zip(temp1, temp2)]
        neuron.act = sig(fsum(smlist))
        print("hit rec, current neuron act: {0} ... current neuron layer: {1}".format(neuron.act, neuron.lyr))

        # or , potentially obnoxious one-liner...

        # neuron.act = fsum([a * b for a,b in zip(neuron.weights, [i[0] for i in network.contents[neuron.lyr -1]])])


def sig(input):
    return (1 / (1 + exp(-input)))

#

## SHOULD DEVELOP A VALIDITY CHECK FUNCTION THAT
## VERIFIES INPUTS TO THE NETWORK ARE VALID!

#
## TEST & I/O FUNCTIONS SECTION
# this code will be run intermediate to make sure things are generated correctly

tr1 = [[0,0,0,1], 1]
def gentest():
#    dflt = str(input("default net or specify ('YES' or 'NO', +return): "))
    dflt = "YES"
    if dflt == "YES":
        hl = 2
        lsz = 4
        isz = 4
        osz = 4
        network = createnet(hl, lsz, isz, osz)
        print("")
        print("{2}-neuron input, network with {0} hidden layers (each with {1} neurons), {3}-neuron output, created.\n"\
              .format(hl,lsz,isz,osz))

    elif dflt == "NO":
        hl = int(input("how many hidden layers (+return): "))
        lsz = int(input("how many neurons in the hidden layers (+return): "))
        isz = int(input("how many neurons in the input layer (+return): "))
        osz = int(input("how many neurons in the output layer (+return): "))

        network = createnet(hl, lsz, isz, osz)

#    opt0 = str(input("display network? ('YES' or 'NO', +return): "))
    opt0 = "YES"
    if opt0 == "YES":
        print("")
        dispnet(network)

    opt1 = str(input("pass in tr1? ('YES' or 'NO' + return): "))
    if opt1 == "YES":
        fprop(network, tr1)
        print("output layer: \n")
        for i in (network.contents[network.hl+1]):
            print(i.act)
            print("")
        print("")
        dispnet(network)
    else:
        print('operation terminated')

def dispnet(network):

    for layer in network.contents:
        print( "layer #{0}".format(int(network.contents.index(layer))))
        print("")
        for neuron in layer:
            print( "neuron #{}".format(int(layer.index(neuron))))
            print("activation:{0} ... weights:{1} ... layer (sanity check):{2}".format(neuron.act, neuron.weights, neuron.lyr))
        print("")

gentest()