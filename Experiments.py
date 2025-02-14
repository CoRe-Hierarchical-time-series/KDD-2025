from abc import ABC, abstractmethod
import coherency
from utils import * 
from constants import * 
from models import *

import pandas as pd
import properscoring as ps

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import tqdm


def validation_loss(model, X_val, y_val):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        pred = model(X_val, test=True)
        return mse_loss(y_val, pred).item()

def generate_samples(model, x, num_samples=1000):
    model.train() # needed to keep dropout 
    with torch.no_grad():  
        samples = []
        for _ in range(num_samples):
            sample = model(x, test=True)
            samples.append(sample)
        
        samples = torch.stack(samples, dim=0)
    
    return samples
        
class Metrics:
    def __init__(self, aggregation_mat, distributional=False): 
        self.aggregation_mat = aggregation_mat
        self.metrics = []
        self.names = []
        self.distributional = distributional
    def run_metrics(self, network, X, y): 
        results = pd.DataFrame()
        metrics = [(getattr(self, method), method) for method in dir(self) if callable(getattr(self, method)) 
                                                                       and not method.startswith("__") 
                                                                       and not method.startswith("run_metrics")]
        if self.distributional:
            all_pred = generate_samples(network, X, num_samples=100) 
            pred = all_pred.mean(0)
        else:
            pred = network(X)
        
        for metric, name in metrics:
            if name == "crps":
                if self.distributional:
                    res, name = metric(network, all_pred, y)
                else: continue
            else:
                res, name = metric(network, pred, y)
            results[name] = res
        return results
    
    def mse_by_level(self, _, pred, y): 
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        mse_by_level = []
        for s in sizes: 
            keep = subtree_size == s
            err = mse_loss(y.to(device)[:, keep].float(), pred[:, keep].to(device).float())
            mse_by_level.append(err.item())
        return mse_by_level, "MSE"
    
    def wmape_by_level(self, _, pred, y): 
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        wmape_by_level = []
        for s in sizes: 
            keep = subtree_size == s
            err = calculate_wmape(y.to(device)[:, keep].float(), pred[:, keep].to(device).float())
            wmape_by_level.append(err.item())
        return wmape_by_level, "WMAPE"
    
    def coherency_levels(self, _, pred, y): 
        coherency_loss = torch.abs(coherency_metric(pred, self.aggregation_mat)).mean(dim=1)
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        coh = []
        for s in sizes: 
            keep = subtree_size == s
            c = coherency_loss[keep].mean()
            coh.append(c.item())
        return coh, "Coherency"
    
    def crps(self, _, samples, y):
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        crps_by_level = [] 
        for s in sizes: 
            keep = subtree_size == s
            cur_y = y[:, keep]
            cur_samples = samples[:,:,keep]
            
            crps_scores = []
            for series in range(cur_y.shape[1]):
                score = ps.crps_ensemble(cur_y[:,series].detach().cpu().numpy(), cur_samples[:, :, series].T.detach().cpu().numpy())
                crps_scores.append(score)
                if series > 20: break
            crps_scores = np.array(crps_scores)
            crps_by_level.append(crps_scores.mean())
        return crps_by_level, "CRPS"
    
def train(network, data_loader, X_val, y_val, params, aggregation_mat=None):
    network.train()
    early_stopping = EarlyStopping(patience=100, verbose=False)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=params['lr'])

    n_epochs = params['n_epochs']
    losses = [1]
    c_losses = [1]
    val_losses = []
    for epoch in tqdm.tqdm(range(n_epochs)):  # loop over the dataset multiple times
        inputs, targets = next(iter(data_loader))

        optimizer.zero_grad()
        network.train()

        inputs = inputs
        targets = targets
                
        if params.get('coherency_loss', False):
            pred = network(inputs.float())
            c_loss = coherency.coherency_loss(network, aggregation_mat)
        elif params.get('profhit', False):
            pred = network(inputs.float())
            c_loss = coherency.coherency_data_loss(pred, aggregation_mat)
        elif params.get('jsd', False): 
            pred, mu, logstd = network(inputs.float())
            c_loss = coherency.jsd_loss(mu, logstd, aggregation_mat, targets)
        else: 
            pred = network(inputs.float())
            c_loss = torch.tensor([0]).to(device)

        loss = mse_loss(pred, targets.float())
        total_loss = loss * np.mean(c_losses) / np.mean(losses) + c_loss * params.get('coherency_weight', 0)
        # total_loss = loss + c_loss * params['coherency_weight']
                
        total_loss.backward() 
        optimizer.step()
        
        losses.append(loss.item())
        c_losses.append(c_loss.item())
        
        val_loss = validation_loss(network, X_val, y_val)
        early_stopping(val_loss, network)
  
        val_losses.append(val_loss)

    # network.load_state_dict(torch.load('checkpoint.pt'))
    return losses, c_losses, val_losses



class Experiment(ABC): 
    def __init__(self, aggregation_mat, params = {'lr':1e-4, 'n_epochs':400}):
        self.aggregation_mat = torch.tensor(aggregation_mat).float().to(device)
        self.full_agg = format_aggregation_matrix(aggregation_mat).float().to(device)
        self.params = params
        self.train_split = self.params['train_split']
        self.val_split = self.params['val_split']
        self.context_window = self.params['context_window']
        
        self.network = None 
        self.distributional = False
        
    def run(self, data):
        X_train, y_train, X_val, y_val, X_test, y_test = self.make_data(data)
        batch_size = X_train.shape[0]
        
        train_dataloader = DataLoader(TensorDataset(X_train.to(device).float(), y_train.to(device).float()), batch_size=batch_size, shuffle=True)
        val_dataloader   = DataLoader(TensorDataset(X_val.to(device).float()  , y_val.to(device).float()  ), batch_size=batch_size, shuffle=True)
        test_dataloader  = DataLoader(TensorDataset(X_test.to(device).float() , y_test.to(device).float() ), batch_size=batch_size, shuffle=True)

        losses = train(self.network, train_dataloader, X_val.to(device).float(), y_val.to(device).float(), self.params, self.full_agg)

        metrics = Metrics(self.full_agg, self.distributional)
        return metrics.run_metrics(self.network, X_test.to(device).float(), y_test.to(device).float()), losses
        
        
    # @abstractmethod
    def make_data(self, data): 
        return get_data(data, self.train_split, self.val_split, self.context_window)
        
        
class BaseModel(Experiment): 
    def __init__(self, aggregation_mat, params): 
        super().__init__(aggregation_mat, params)
        
        network = RNN(self.aggregation_mat.to(device), self.params)
        self.network = network.to(device)
    
class JSDDistribution(Experiment): 
    def __init__(self, aggregation_mat, params): 
        super().__init__(aggregation_mat, params)
        
        params['jsd'] = True 
        self.network = DistForecast(torch.tensor(aggregation_mat).to(device).float(), params, project=False).to(device)
        self.distributional = True

class ProjectDistribution(Experiment): 
    def __init__(self, aggregation_mat, params): 
        super().__init__(aggregation_mat, params)
        
        self.network = DistForecast(torch.tensor(aggregation_mat).to(device).float(), params, project=True).to(device)
        self.distributional = True

        
class DropoutDistribution(Experiment): 
    def __init__(self, aggregation_mat, params): 
        super().__init__(aggregation_mat, params)
        
        self.network = DropForecast(torch.tensor(aggregation_mat).to(device).float(), params).to(device)
        self.distributional = True

        
class VAEDistribution(Experiment): 
    def __init__(self, aggregation_mat, params): 
        super().__init__(aggregation_mat, params)
        
        self.network = VAEForecast(torch.tensor(aggregation_mat).to(device).float(), params).to(device)
        self.distributional = True


def cv(model_class, base_agg_mat, data, params):
    best = None
    all_res = []
    best_w = None
    for w in [1e-4, 1e-3, 1e-2]:
        params['coherency_weight'] = w
        model = model_class(base_agg_mat, params)
        res, losses = model.run(utils.add_noise(data.copy(), base_agg_mat, params['noise'], seed=0))
        
        all_res.append((w, res))
        if best is None or res["WMAPE"].mean() < best["WMAPE"].mean(): 
            best = res
            best_w = w
    return best_w

def repeat_noise_exp(model_class, base_agg_mat, data, params):
    results = []
    for i in range(params['n_runs']):
        print("run", i)
        model = model_class(base_agg_mat, params)
        res, losses = model.run(utils.add_noise(data.copy(), base_agg_mat, params['noise'], seed=i))
        results.append(res.values)
    return np.array(results), res.columns, losses

def repeat_exp(model_class, base_agg_mat, data, params, coherency_loss=False, profhit_loss=False, project=False):
    params['n_series'] = data.shape[1]
    params['coherency_loss'] = coherency_loss
    params['profhit_loss'] = profhit_loss
    params['project'] = project

    results = []
    all_losses = []
    for i in range(params['n_runs']):
        print("run", i)
        
        model = model_class(base_agg_mat, params)
        res, losses = model.run(data)
        results.append(res.values)
        all_losses.append(losses)
        
    return np.array(results), res.columns, all_losses

def get_mean(results, names): 
    return pd.DataFrame(columns=names, data=results.mean(axis=0))
def get_std(results, names): 
    return pd.DataFrame(columns=names, data=results.std(axis=0))