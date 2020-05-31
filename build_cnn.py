
import tflearn
from tflearn.layers.core import input_data, dropout,flatten, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def create_model():
        print("Building CNN")

        network = input_data(shape=[None, 48, 48, 1])
        #print("Input Data",network.shape[1:])
        network = conv_2d(network, 32, 3, padding='SAME', activation='relu')
        network = max_pool_2d(network, 2, strides=2, padding='SAME')

        network = conv_2d(network, 64, 3, padding='SAME', activation='relu')
        network = max_pool_2d(network, 2, strides=2, padding='SAME')

        network = conv_2d(network, 64, 3, padding='SAME', activation='relu')
        network = max_pool_2d(network, 2, strides=2, padding='SAME')

        network = conv_2d(network, 128, 3, padding='SAME', activation='relu')
        network = flatten(network)
        network = fully_connected(network, 3072, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 7, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        print('MODEL CREATED ....')
        return network
