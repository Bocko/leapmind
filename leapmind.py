import random
import numpy as np
from mnist import MNIST

#################################################################################
# Possible improvements:
# * Better cost function (cross entropy?)
# * Initialize the synaptic weights with better values?
# * Improve the SGD
# * Use a heuristic (e.g. genetic algorithm) to tune the hyperparameters
#   (e.g. number of layers, neurons per layer, split_size, learning rate, ...)
# * Deep learning?
#
# Resources:
# https://iamtrask.github.io/2015/07/12/basic-python-network/
# https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
# http://natureofcode.com/book/chapter-10-neural-networks/
# http://sebastianruder.com/optimizing-gradient-descent/index.html
#################################################################################

def init_synapses(layer_sizes):
    """
    Initializes biases and synaptic weights
    """
    biases = [np.random.randn(i, 1) for i in layer_sizes[1:]]
    synaptic_weights = [np.random.randn(j, i) for i, j in zip(layer_sizes[:-1], layer_sizes[1:])]

    return biases, synaptic_weights

def sigmoid(x, deriv=False):
    """
    Simple sigmoid function
    """
    if deriv:
        return x*(1-x)

    return 1/(1+np.exp(-x))

def feedforward(x, biases, synaptic_weights):
    """
    Feed forward
    """
    layer = x   # First layer
    layer_list = [x]
    for bias, weight in zip(biases, synaptic_weights):  # Update next layers
        layer = sigmoid(np.dot(weight, layer) + bias)
        layer_list.append(layer)

    return layer_list

def backpropagation(y, layer_list, biases, synaptic_weights):
    """
    Back-propagation
    """
    layer_nb = len(synaptic_weights) + 1
    bp_delta_list = [np.zeros(bias.shape) for bias in biases]
    bp_error_list = [np.zeros(weight.shape) for weight in synaptic_weights]
    error = layer_list[-1] - y  # TODO: improve this simple error function
    delta = error * sigmoid(layer_list[-1], deriv=True)
    bp_delta_list[-1] = delta   # Last layer delta
    bp_error_list[-1] = np.dot(delta, layer_list[-2].T) # Last layer error
    for i in range(layer_nb-2, 0, -1):  # Update previous layer delta/error
        layer = layer_list[i]
        delta = np.dot(synaptic_weights[i].T, delta) * sigmoid(layer, deriv=True)
        bp_delta_list[i-1] = delta
        bp_error_list[i-1] = np.dot(delta, layer_list[i-1].T)

    return bp_delta_list, bp_error_list

def propagation(x, y, biases, synaptic_weights):
    """
    Feed forward and back-propagation
    """
    # Feed forward
    layer_list = feedforward(x, biases, synaptic_weights)
    # Back-propagation
    bp_delta_list, bp_error_list = backpropagation(y, layer_list, biases, synaptic_weights)

    return bp_delta_list, bp_error_list

def synapse_update(split, rate, biases, synaptic_weights):
    """
    Updates the synaptic weights
    """
    delta_list = [np.zeros(bias.shape) for bias in biases]
    error_list = [np.zeros(weight.shape) for weight in synaptic_weights]
    for x, y in split:
        bp_delta_list, bp_error_list = propagation(x, y, biases, synaptic_weights)
        delta_list = [delta + bp_delta for delta, bp_delta in zip(delta_list, bp_delta_list)]
        error_list = [error + bp_error for error, bp_error in zip(error_list, bp_error_list)]
    synaptic_weights = [weight - rate*error/len(split) for weight, error in zip(synaptic_weights, error_list)]
    biases = [bias - rate*delta/len(split) for bias, delta in zip(biases, delta_list)]

    return biases, synaptic_weights

def train(training_set, iter_nb, split_size, rate, biases, synaptic_weights, test_set=None):
    """
    Stochastic Gradient Descent training, by splitting the dataset
    """
    for i in range(iter_nb):
        random.shuffle(training_set)
        splits = [training_set[j:j+split_size] for j in range(0, len(training_set), split_size)]
        for split in splits:
            biases, synaptic_weights = synapse_update(split, rate, biases, synaptic_weights)
        if test_set:
            print("Iter " + str(i+1) + '/' + str(iter_nb) + ',', "Precision " + str(evaluate(test_set, biases, synaptic_weights)) + '/' + str(len(test_set)))
        else:
            print("Iter " + str(i+1) + '/' + str(iter_nb))

def evaluate(test_set, biases, synaptic_weights):
    """
    Evaluates the number of correct results compared to test_set
    """
    y_pred_list = []
    y_ref_list = []
    for x, y in test_set:
        y_pred = sigmoid(np.dot(synaptic_weights[0], x) + biases[0])  # Init
        for bias, weight in zip(biases[1:], synaptic_weights[1:]):
            y_pred = sigmoid(np.dot(weight, y_pred) + bias)
        y_pred_list.append(np.argmax(y_pred))
        y_ref_list.append(y)

    return sum([y_pred == y_ref for y_pred, y_ref in zip(y_pred_list, y_ref_list)])

def load_data(dir_name):
    """
    Load the data, reshape, flatten and normalize it
    """
    mnistdata = MNIST(dir_name)
    images_train, labels_train = mnistdata.load_training()
    images_test, labels_test = mnistdata.load_testing()
    pix_nb = len(images_train[0])
    # Reshape to np.array, flatten and normalize
    images_train_data = np.array([np.reshape(image, (pix_nb, 1)).astype('float32') / 255 for image in images_train])
    labels_train_data = [val_to_vect(val) for val in labels_train]
    training_set = list(zip(images_train_data, labels_train_data))
    images_test_set = np.array([np.reshape(image, (pix_nb, 1)).astype('float32') / 255 for image in images_test])
    test_set = list(zip(images_test_set, labels_test))

    return training_set, test_set

def val_to_vect(val):
    """
    Convert the digit value into a 10-bit vector with 1 at the position of the value and 0 elsewhere
    """
    vect = np.zeros((10, 1))
    vect[val] = 1

    return vect

def main():
    SEED = 32
    random.seed(SEED)
    np.random.seed(SEED)
    dir_name = 'files'
    training_set, test_set = load_data(dir_name)
    layer_sizes = [784, 40, 10]
    iter_nb = 30
    split_size = 10
    rate = 4
    biases, synaptic_weights = init_synapses(layer_sizes)
    train(training_set, iter_nb, split_size, rate, biases, synaptic_weights, test_set=test_set)

if __name__ == "__main__":
    main()
