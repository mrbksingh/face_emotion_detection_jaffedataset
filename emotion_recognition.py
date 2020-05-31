from parameters import *
from os.path import join

class Dataset:
    name = DATASET_CSV_FILENAME

class Training:
    model = SAVE_MODEL_FILENAME
    save_model = True
    model_path = join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)

DATASET = Dataset()
TRAINING = Training()
