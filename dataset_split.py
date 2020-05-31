
from parameters import *
from emotion_recognition import DATASET
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(validation=False, test=False):  
    train_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == DATASET_CSV_FILENAME:
        
        # load train set
        images = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))         
        images = images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])        
        labels = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME))
        labels = labels.reshape([-1, len(EMOTIONS)])
        X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.8, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.7, test_size=0.3, random_state=42)

        train_dict['X'] = X_train
        train_dict['Y'] = y_train
        validation_dict['X'] = X_val
        validation_dict['Y'] = y_val
        test_dict['X'] = X_test
        test_dict['Y'] = y_test

        

        if not validation and not test:
            print("loading dataset " +DATASET.name+ "...")
            return train_dict
        elif not test:
            print("loading dataset " +DATASET.name+ "...")
            return train_dict, validation_dict
        else: 
            print("loading dataset " +DATASET.name+ "...")
            return train_dict, validation_dict, test_dict
    else:
        print( "Unknown dataset")
        exit()

