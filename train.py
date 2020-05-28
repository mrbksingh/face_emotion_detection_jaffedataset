from __future__ import division, absolute_import
import argparse
import tensorflow as tf
import tflearn
import time
import os
from os.path import isfile, join
from dataset_split import load_data
from parameters import *
from emotion_recognition import DATASET, NETWORK, TRAINING
from build_cnn import create_model



def train(train_model=True):
        
        if train_model:
            data, validation = load_data(validation=True) 
        else:
            data, validation, test = load_data(validation=True, test=True)

        with tf.Graph().as_default():
            print("Creating Model .....")
            network = create_model()
            model = tflearn.DNN(
                network,
                checkpoint_path = SAVE_DIRECTORY + '/FD_Model',
                max_checkpoints=1,
                tensorboard_verbose=0,
                tensorboard_dir="logs"
            )

            if train_model:
                print('----------------------------------------------------------------------')
                print('----------------------------------------------------------------------')
                print("Training started....")
                print('Details :')
                print('  - Emotions : {}'.format(len(EMOTIONS)))
                print('  - Model name : {}'.format(NETWORK.model))
                print('  - Dataset name : {}'.format(DATASET.name))
                start_time = time.time()
                model.fit(
                            data['X'], data['Y'],
                            validation_set=(validation['X'], validation['Y']),
                            n_epoch=15,
                            batch_size=128,
                            shuffle = True,
                            show_metric=True,
                            snapshot_step=500,
                            snapshot_epoch=True,
                )
                validation['X2'] = None
                training_time = time.time() - start_time
                print("Training time = {0:.1f} sec".format(training_time))

                if TRAINING.save_model:
                       print("Saving model...")
                       model.save(TRAINING.model_path)
                       if not(isfile(TRAINING.model_path)) and \
                            isfile(TRAINING.model_path + ".meta"):
                            os.rename(TRAINING.model_path + ".meta", TRAINING.model_path)
                print("evaluating...")
                validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'])
                print("Validation Accuracy = {0:.1f}".format(validation_accuracy*100))
                return validation_accuracy
            
            else:
                # Test Model
                
                print("Evaluating model....")
                print("Loading pretrained model....")

                if isfile(TRAINING.model_path):
                    model.load(TRAINING.model_path)
                else:
                    print("Error: file not found")
                    exit()
        
                print("--------------------------------------------------")
                validation['X2'] = None
                test['X2'] = None
                start_time = time.time()
                validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'])
                print("Validation Accuracy = {0:.1f}".format(validation_accuracy*100))
                test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'])
                print( "  - test accuracy = {0:.1f}".format(test_accuracy*100))
                print( "  - evalution time = {0:.1f} sec".format(time.time() - start_time))
                
                return test_accuracy


def evaluate(model, X, X2, Y):
    accuracy = model.evaluate(X, Y)
    return accuracy[0]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e","--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
    train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
    train(train_model=False)
