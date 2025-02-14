import torch 
import numpy as np
from constants import *

def coherency_loss(network, aggregation_mat): 
    repeated_bias = network.last_layer.bias.repeat(network.last_layer.weight.shape[0], 1)
    return torch.norm(aggregation_mat @ network.last_layer.weight - network.last_layer.weight) + torch.norm(
        aggregation_mat @ network.last_layer.bias - network.last_layer.bias) 

def coherency_data_loss(predictions, aggregation_mat):
    err = predictions.T - aggregation_mat @ predictions.T
    return torch.norm(err)

class Projection: 
    def __init__(self, S):
        self.M, self.A = self.create_M(S)
        self.M.to(device)

    def create_M(self, S): 
        # creates the projection matrix M given the aggregation matrix S
        m, m_K = S.shape
        m_agg = m-m_K

        # The top `m_agg` rows of the matrix `S` give the aggregation constraint matrix.
        S_agg = S[:m_agg, :]
        A = torch.hstack((torch.eye(m_agg).to(device).float(), -S_agg))

        M = torch.eye(m).to(device).float() - A.T @ torch.inverse(A @ A.T) @ A  

        return M.float(), A.float()

    def project(self, y): 
        return torch.matmul(y, self.M)

    
def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    # print(mu_diff.abs().max(), var1.abs().max(), var2.abs().max())
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2

def jsd_loss(mu, logstd, hmatrix, data):
    train_means = torch.tensor(torch.mean(data, dim=0), device=device).float()
    train_std = torch.tensor(torch.std(data, dim=0), device=device).float()
    
    lhs_mu = ((hmatrix @ (mu * train_std + train_means).T).T - train_means) / (train_std)
    lhs_var = (hmatrix @ ((torch.exp(2.0 * logstd) * (train_std ** 2)).T)).T / (train_std ** 2)
    ans = torch.nan_to_num(jsd_norm(mu, lhs_mu, (2.0 * logstd).exp(), lhs_var))
    # print(ans)
    return ans.mean()