
from __future__ import division, absolute_import
import tflearn
import time
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum, Adam
from parameters import *
from emotion_recognition import DATASET,NETWORK
from os.path import join
import argparse

def create_model(optimizer=NETWORK.optimizer, 
                   optimizer_param=NETWORK.optimizer_param, 
                   learning_rate=NETWORK.learning_rate, 
                   learning_rate_decay=NETWORK.learning_rate_decay,
                   decay_step=NETWORK.decay_step):
    if NETWORK.model == 'A':
        return create_model_A(optimizer, optimizer_param, learning_rate, learning_rate_decay, decay_step)
    elif NETWORK.model == 'B':
        return create_model_B(optimizer, optimizer_param, learning_rate, learning_rate_decay, decay_step)
    else:
        print("Error: Please select a model" + str(NETWORK.model))
        exit()

def create_model_A(optimizer=NETWORK.optimizer, 
                   optimizer_param=NETWORK.optimizer_param, 
                   learning_rate=NETWORK.learning_rate, 
                   learning_rate_decay=NETWORK.learning_rate_decay,
                   decay_step=NETWORK.decay_step):
        print("Building CNN")
        network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1], name='input_A')
        network = conv_2d(network, 64, 5, activation='relu')

        if NETWORK.use_batchnorm_after_conv_layers:
            network = batch_normalization(network)

        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 64, 5, activation='relu')

        if NETWORK.use_batchnorm_after_conv_layers:
            network = batch_normalization(network)

        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 128, 4, activation='relu')

        if NETWORK.use_batchnorm_after_conv_layers:
            network = batch_normalization(network)

        network = dropout(network, 0.956)
        network = fully_connected(network, 1024, activation='relu')
        
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            network = batch_normalization(network)

        network = fully_connected(network, len(EMOTIONS), activation='softmax')

        if optimizer == 'momentum':
            optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param, lr_decay=learning_rate_decay, decay_step=decay_step)
        elif optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
        else:
            print( "Unknown optimizer: {}".format(optimizer))

        network = regression(network,
            optimizer=optimizer,
            loss='categorical_crossentropy',
            learning_rate=learning_rate,            
            name='output'
        )
        print('MODEL A CREATED ....')
        return network

def create_model_B(optimizer=NETWORK.optimizer, 
                   optimizer_param=NETWORK.optimizer_param, 
                   learning_rate=NETWORK.learning_rate, 
                   learning_rate_decay=NETWORK.learning_rate_decay,
                   decay_step=NETWORK.decay_step):
        print('[+] Building CNN')
        network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1], name='input_A')

        network = conv_2d(network, 64, 3, activation='relu')
        if NETWORK.use_batchnorm_after_conv_layers:
            network = batch_normalization(network)

        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 128, 3, activation='relu')

        if NETWORK.use_batchnorm_after_conv_layers:
            network = batch_normalization(network)

        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 256, 3, activation='relu')

        if NETWORK.use_batchnorm_after_conv_layers:
            network = batch_normalization(network)

        network = max_pool_2d(network, 3, strides=2)
        network = dropout(network, 0.956)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.956)
        network = fully_connected(network, 1024, activation='relu')
        
        
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            network = batch_normalization(network)

        network = fully_connected(network, len(EMOTIONS), activation='softmax')


        if optimizer == 'momentum':
            optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param, lr_decay=learning_rate_decay, decay_step=decay_step)
        elif optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
        else:
            print( "Unknown optimizer: {}".format(optimizer))

        network = regression(network,
            optimizer=optimizer,
            loss='categorical_crossentropy',
            learning_rate=learning_rate,
            name='output'
        )
        print('MODEL B CREATED ....')
        return network

