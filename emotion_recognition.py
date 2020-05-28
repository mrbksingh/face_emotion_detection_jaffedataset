from parameters import *
from os.path import join

class Dataset:
    name = DATASET_CSV_FILENAME
    
class Network:
    model = 'B'
    use_batchnorm_after_conv_layers = True
    use_batchnorm_after_fully_connected_layers = False
    learning_rate = 0.016
    learning_rate_decay = 0.864
    decay_step = 50
    optimizer = 'momentum'      # {'momentum' , 'adam'}
    optimizer_param = 0.95
    
class Training:
    save_model = True
    model_path = join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
