from tqdm import tqdm
import numpy as np
import time
import os
""" Torch """
import torch
from data_ks_preprocess import bfs_dataset
from torch.utils.data import DataLoader, Dataset
""" Utilities """


from functools import partial

print = partial(print, flush=True)
import warnings

def pca_svd(X):
    """ Perform PCA using SVD"""
    X = np.asarray(X)
    n_samples, n_features = X.shape

    # Center the data
    mean = np.mean(X, axis=0)
    center = X-mean

    U,s,Vh = np.linalg.svd(center/np.sqrt(n_samples), full_matrices=False)


    variances = s**2
    pcs = Vh

    return pcs, variances, mean

def project_pca(X_centered, pcs, latent_dim):
    """Projects centered data onto the top principal components."""
    pcs_reduced = pcs[:latent_dim, :] # Take top 'latent_dim' components
    # Project: (n_samples, n_features) @ (n_features, latent_dim) -> (n_samples, latent_dim)
    projected = X_centered @ pcs_reduced.T # Note the transpose
    return projected


def reconstruct_pca(projected_data, pcs, latent_dim, mean):
    """Reconstructs data from the projected representation."""
    pcs_reduced = pcs[:latent_dim, :] # Take top 'latent_dim' components
    # Reconstruct: (n_samples, latent_dim) @ (latent_dim, n_features) -> (n_samples, n_features)
    reconstructed_centered = projected_data @ pcs_reduced
    reconstructed = reconstructed_centered + mean # Add mean back
    return reconstructed


def complex_mse(y_true, y_pred):
    """Calculates Mean Squared Error for complex numbers."""
    return np.mean(np.abs(y_true - y_pred)**2)

def evaluate_pca_reconstruction_per_trajectory(data_loader, latent_dim):
    """
    Evaluates PCA reconstruction error using SVD for complex data, 
    fitting PCA to each trajectory individually.

    Args:
        data_loader: An iterable that yields individual trajectories. 
                     Each trajectory should be a NumPy array of shape [time_step, spatial_dimension].
        latent_dim: The number of principal components (latent dimension) to use for PCA.

    Returns:
        float: The mean squared error averaged over all trajectories in the data_loader.
    """
    all_mse = []
    print(f"[evaluate_pca] Starting evaluation with latent_dim = {latent_dim}")

    for i, trajectory in enumerate(data_loader):
        # Ensure trajectory is a NumPy array
        trajectory = np.asarray(trajectory).squeeze()
            
        if trajectory.ndim != 2:
            print(f"[evaluate_pca] Warning: Skipping trajectory {i} due to unexpected shape {trajectory.shape}")
            continue
            
        n_samples, n_features = trajectory.shape
        effective_latent_dim = min(latent_dim, n_samples, n_features) # Cannot have more components than samples or features

        if effective_latent_dim != latent_dim:
             print(f"[evaluate_pca] Warning: Reducing latent_dim from {latent_dim} to {effective_latent_dim} for trajectory {i} due to data shape {trajectory.shape}")
        if i == 0:
            print(f"[evaluate_pca] Processing trajectory {i} with shape {trajectory.shape}...")

        # 1. Fit PCA using SVD
        pcs,variances, mean = pca_svd(trajectory)
            
            # Center the data for projection/reconstruction
        centered_trajectory = trajectory - mean

            # 2. Project to latent space

        projected_data = project_pca(centered_trajectory, pcs, effective_latent_dim)

            # 3. Reconstruct from latent space
        reconstructed_trajectory = reconstruct_pca(projected_data, pcs, effective_latent_dim, mean)

            # 4. Calculate Complex MSE
        mse = complex_mse(trajectory, reconstructed_trajectory)
        all_mse.append(mse)
        if i % 100 == 0:
            print(f"[evaluate_pca] Trajectory {i} MSE: {mse:.6f}")

    mean_mse = np.mean(all_mse)
    return mean_mse

if __name__ == "__main__":
    # Example usage
    latent_dim = 16
    #data_path = 'data/data1.npy'
    ks_dataset = bfs_dataset()
    data_loader = DataLoader(ks_dataset, batch_size=1, shuffle=False)

    average_reconstruction_error = evaluate_pca_reconstruction_per_trajectory(data_loader, latent_dim)
    print(f"Average reconstruction error for latent dimension {latent_dim}: {average_reconstruction_error:.6f}")




'''
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

        #self.channels = params['channels']
        #self.Dz, self.Dy, self.Dx = utils.getChannels(self.channels, params)

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
        #assert len(np.shape(data)) == 2 + 1 + self.channels
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
        print(
            "[dimred] Saving dimensionality reduction results with name: {:}".
            format(model_name_dimred))

        print("[dimred] Recording time...")
        self.total_training_time = time.time() - self.start_time

        print("[dimred] Total training time is {:}".format(
            utils.secondsToTimeStr(self.total_training_time)))

        self.memory = utils.getMemory()
        print("[dimred] Script used {:} MB".format(self.memory))

        data = {
            "params": self.params,
            "model_name": self.model_name,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "dimred_model": self.dimred_model,
        }
        fields_to_write = [
            "memory",
            "total_training_time",
        ]
        if self.write_to_log == 1:
            logfile_train = self.saving_path + self.logfile_dir + model_name_dimred + "/train.txt"
            print("[dimred] Writing to log-file in path {:}".format(
                logfile_train))
            utils.write_to_logfile(self, logfile_train, data, fields_to_write)

        data_folder = self.saving_path + self.model_dir + model_name_dimred
        os.makedirs(data_folder, exist_ok=True)
        data_path = data_folder + "/data"
        utils.saveData(data, data_path, self.params["save_format"])

    def load(self):
        model_name_dimred = self.createModelName()
        print(
            "[dimred] Loading dimensionality reduction from model: {:}".format(
                model_name_dimred))
        data_path = self.saving_path + self.model_dir + model_name_dimred + "/data"
        print("[dimred] Datafile: {:}".format(data_path))
        try:
            data = utils.loadData(data_path, self.params["save_format"])
        except Exception as inst:
            raise ValueError(
                "[Error] Dimensionality reduction results {:s} not found.".
                format(data_path))
        self.dimred_model = data["dimred_model"]
        del data
        return 0

    def test(self):
        if self.load() == 0:
            testing_modes = self.getTestingModes()
            test_on = []
            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")
            for set_ in test_on:
                common_testing.testModesOnSet(self,
                                              set_=set_,
                                              testing_modes=testing_modes)
        return 0

    def getTestingModes(self):
        return ["dimred_testing"]

    def plot(self):
        if self.write_to_log:
            common_plot.writeLogfiles(self, testing_mode="dimred_testing")
        else:
            print("[dimred] # write_to_log=0. #")

        if self.params["plotting"]:
            common_plot.plot(self, testing_mode="dimred_testing")
        else:
            print("[dimred] # plotting=0. No plotting. #")
        '''