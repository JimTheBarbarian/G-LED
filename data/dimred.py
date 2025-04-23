from tqdm import tqdm
import numpy as np
import time
import os
""" Torch """
import torch
""" Utilities """


from functools import partial

print = partial(print, flush=True)
import warnings



class dimred():
    def __init__(self, params):
        super(dimred, self).__init__()
        # Starting the timer
        self.start_time = time.time()
        # The parameters used to define the model
        self.params = params.copy()

        # The system to which the model is applied on
        self.system_name = params["system_name"]
        # Checking the system name
        assert (systems.checkSystemName(self))
        # The save format
        self.save_format = params["save_format"]

        ##################################################################
        # RANDOM SEEDING
        ##################################################################
        self.random_seed = params["random_seed"]

        # Setting the random seed
        np.random.seed(self.random_seed)

        self.data_path_train = params['data_path_train']
        # The path of the training data
        self.data_path_val = params['data_path_val']
        # General data path (scaler data/etc.)
        self.data_path_gen = params['data_path_gen']
        # The path of the test data
        self.data_path_test = params['data_path_test']
        # The path to save all the results
        self.saving_path = params['saving_path']
        # The directory to save the model (inside saving_path)
        self.model_dir = params['model_dir']
        # The directory to save the figures (inside saving_path)
        self.fig_dir = params['fig_dir']
        # The directory to save the data results (inside saving_path)
        self.results_dir = params['results_dir']
        # The directory to save the logfiles (inside saving_path)
        self.logfile_dir = params["logfile_dir"]
        # Whether to write a log-file or not
        self.write_to_log = params["write_to_log"]

        # Whether to display in the output (verbocity)
        self.display_output = params["display_output"]

        # The number of IC to test on
        self.num_test_ICS = params["num_test_ICS"]

        # The prediction horizon
        self.prediction_horizon = params["prediction_horizon"]

        self.input_dim = params['input_dim']

        self.channels = params['channels']
        self.Dz, self.Dy, self.Dx = utils.getChannels(self.channels, params)

        ##################################################################
        # SCALER
        ##################################################################
        self.scaler = params["scaler"]

        ##################################################################
        # DimRed parameters (PCA/DiffMaps)
        ##################################################################
        self.latent_state_dim = params["latent_state_dim"]
        self.dimred_method = params["dimred_method"]

        if self.dimred_method == "pca":
            """ PCA has no hyper-parameters """
            pass

    def getKeysInModelName(self):
        keys = {
            'scaler': '-scaler_',
            'dimred_method': '-METHOD_',
        }

        if self.dimred_method == "pca":
            keys.update({
                'latent_state_dim': '-LD_',
            })

        return keys

    def getModelName(self):
        keys = self.getKeysInModelName()
        str_ = "DimRed"
        for key in keys:
            key_to_print = utils.processList(self.params[key])
            str_ += keys[key] + "{:}".format(key_to_print)
        return str_
    

    def applyDimRed(self):
        assert len(np.shape(data)) == 2 + 1 + self.channels
        shape_ = np.shape(data)
        data = np.reshape(data, (shape_[0]*shape_[1], -1))
        data = self.dimred_model.transform(data)
        data = np.reshape(data, (shape_[0], shape_[1], -1))
        return data

    def applyInverseDimRed(self, data):
        assert len(np.shape(data)) == 3
        """ Use the dimensionality reduction method to lift the projected data """
        shape_ = np.shape(data)
        data = np.reshape(data, (shape_[0] * shape_[1], -1))
        data = self.dimred_model.inverse_transform(data)
        data = np.reshape(data,
                          (shape_[0], shape_[1], *utils.getInputShape(self)))
        return data

    def encodeDecode(self, data):
        latent_state = self.applyDimRed(data)
        output = self.applyInverseDimRed(latent_state)
        return output, latent_state

    def encode(self, data):
        latent_state = self.applyDimRed(data)
        return latent_state

    def decode(self, latent_state):
        data = self.applyInverseDimRed(latent_state)
        return data
    

    def train(self):
        data = []
        batch_size = 1



        for sequence in data_loader_train:
            data.append(sequence)
        
        data = np.array(data)
        from sklearn.decomposition import PCA
        self.dimred_model = PCA(n_components=self.latent_state_dim)
        self.dimred_model.fit(data)

        data_red = self.dimred_model.transform(data)

        self.saveDimRed()

    def saveDimRed(self):
        model_name_dimred = self.getModelName()
        