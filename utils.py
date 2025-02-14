import numpy as np 
import pandas as pd
import torch
import math

from constants import *

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def crps_sample(predictions, observations):
    n_forecasts = predictions.size(1)
    sorted_preds = torch.sort(predictions, dim=1).values
    obs_expanded = observations.unsqueeze(1).expand_as(sorted_preds)
    
    indicator = (sorted_preds <= obs_expanded).float()
    integral = torch.trapz(torch.square(indicator - sorted_preds), dx=1.0/n_forecasts)
    
    return integral

        
def mse_loss(predictions, target): 
    return torch.square(predictions - target).mean()

def wmape_level(y_test_pred, y_test, aggregation_mat): 
    subtree_size = aggregation_mat.sum(dim=1)
    sizes = torch.unique(subtree_size)
    mse_by_level = []
    err_by_level = []
    for s in sizes: 
        keep = subtree_size == s
        err = calculate_wmape(y_test.to(device)[:, keep].float(), y_test_pred[:, keep].to(device).float())
        err_by_level.append(err.item())
    
    return err_by_level

def mse_level(network, y_test_pred, pred, y): 
    subtree_size = aggregation_mat.sum(dim=1)
    sizes = torch.unique(subtree_size)
    mse_by_level = []
    err_by_level = []
    for s in sizes: 
        keep = subtree_size == s
        err = mse_loss(y_test.to(device)[:, keep].float(), y_test_pred[:, keep].to(device).float())
        err_by_level.append(err.item())
    
    return err_by_level

def get_bottom_level(data, aggregation_mat): 
    subtree_size = aggregation_mat.sum(axis=1)
    keep = subtree_size == 1
    return data[:, keep]

def aggregate(y_bottom, aggregation_mat): 
    return (aggregation_mat[:, -y_bottom.shape[1]:] @ y_bottom.T).T

def coherency_metric(predictions, aggregation_mat):
    return predictions.T - aggregation_mat @ predictions.T

def coherency_levels(pred, aggregation_mat): 
    coherency_loss = torch.abs(coherency_metric(pred, aggregation_mat)).mean(dim=1)
    
    subtree_size = aggregation_mat.sum(dim=1)
    sizes = torch.unique(subtree_size)
    coh = []
    for s in sizes: 
        keep = subtree_size == s
        c = coherency_loss[keep].mean()
        coh.append(c.item())
    return coh
        
def calculate_wmape(actual_values, forecasted_values):
    num = torch.sum(torch.abs(actual_values - forecasted_values))
    den = torch.sum(torch.abs(actual_values))
    wmape = num / den
    return wmape

def calculate_RMSE(actual_values, forecasted_values): 
    squared_errors = (actual_values - forecasted_values) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

def format_aggregation_matrix(agg_mat_df): 
    return torch.tensor(np.append(np.zeros((agg_mat_df.shape[0], agg_mat_df.shape[0] - agg_mat_df.shape[1])), agg_mat_df, axis=1), dtype=float)



def load_data(dataset_name): 
    data_file = "{}/data.csv".format(dataset_name)
    hier_file = "{}/agg_mat.csv".format(dataset_name)

    data = pd.read_csv(data_file, index_col=0)
    agg_mat_df = pd.read_csv(hier_file, index_col=0)
    base_agg_mat = agg_mat_df.values

    maximum = np.max(data.values)
    data = (data / maximum).values
    
    return data, base_agg_mat

def make_data(dataset, range_, context_window): 
    # produces X_data and y_data tensors given the dataset 
    X_data = []
    y_data = []
    for i in range_:
        X = dataset[i:i+context_window,:].T
        X_data.append(X)

        y = dataset[i+context_window:i+context_window+1,:].T.flatten()
        y_data.append(y)

    return torch.tensor(np.array(X_data), dtype=float), torch.tensor(np.array(y_data), dtype=float)

def add_noise(data, aggregation_mat, noise, seed=0):
    np.random.seed(seed)
    num_elements = data.shape[1]
    num_zeros = int(num_elements * noise)
    
    subtree_size = aggregation_mat.sum(axis=1)
    keep = subtree_size == 1
    
    indices = np.random.choice(np.arange(num_elements)[keep], size=num_zeros, replace=False)
    data[:,indices] = 1e-4
    # print(data)
    return data
    
def get_data(data, train_split, val_split, context_window):
    n_series = data.shape[1] 
    n_total = data.shape[0]
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    X_train, y_train = make_data(data, range(n_train), context_window)
    X_val, y_val = make_data(data, range(n_train, n_train+n_val), context_window)
    X_test, y_test = make_data(data, range(n_train + n_val,n_total - context_window), context_window)
    
    return X_train, y_train, X_val, y_val, X_test, y_test



class Normal(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = torch.pow(value - self.means, 2)
        log_prob *= -(1 / (2.0 * self.logscales.mul(2.0).exp()))
        log_prob -= self.logscales + 0.5 * math.log(2.0 * math.pi)
        return log_prob

    def sample(self, **kwargs):
        eps = torch.normal(
            float_tensor(self.means.size()).zero_(),
            float_tensor(self.means.size()).fill_(1),
        )
        return self.means + self.logscales.exp() * eps

    def rsample(self, **kwargs):
        return self.sample(**kwargs)