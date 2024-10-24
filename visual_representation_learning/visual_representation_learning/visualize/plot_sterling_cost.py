"""
"""
# TODO: Test code

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Dataset, ConcatDataset
import yaml
from tqdm import tqdm

from scripts.train_naturl_representations import NATURLDataModule
from scripts.models import VisualEncoderModel, CostNet

if __name__ == '__main__':
    
    data_config_path = '/robodata/haresh92/spot-vrl/spot_data/data_config.yaml'
    dm = NATURLDataModule(data_config_path)
    dataset = dm.val_dataset
    
    visual_encoder = VisualEncoderModel(latent_size=128)
    cost_net = CostNet(latent_size=128)
    
    model = nn.Sequential(visual_encoder, cost_net)
    # load weights of model
    model_state_dict = torch.load('/robodata/haresh92/spot-vrl/models/acc_0.99604_20-01-2023-08-13-17_/cost_model.pt')
    model.load_state_dict(model_state_dict)
    model.eval()
    model.cuda()
    
    data = {}
    
    # for patch1, patch2, imu, leg, feet, label, idx in tqdm(dataset):
    #     # convert to torch tensor and add batch dimension
    #     patch1 = torch.tensor(patch1).unsqueeze(0)
    #     patch2 = torch.tensor(patch2).unsqueeze(0)
    #     imu = torch.tensor(imu).unsqueeze(0)
    #     leg = torch.tensor(leg).unsqueeze(0)
    #     feet = torch.tensor(feet).unsqueeze(0)
        
    #     # move to device
    #     patch1 = patch1.cuda()
    #     patch2 = patch2.cuda()
    #     imu = imu.cuda()
    #     leg = leg.cuda()
    #     feet = feet.cuda()
        
    #     # forward pass
    #     cost = model(patch1.float())
    #     cost = cost.detach().cpu().numpy().flatten()[0]
        
    #     if label not in data:
    #         data[label] = []
    #     data[label].append(cost)
        
    #     # if idx > 500:
    #     #     break
        
    # the above code does per sample evaluation. the below code does per batch evaluation
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    for batch in tqdm(dataloader):
        patch1, patch2, imu, leg, feet, label, idx = batch
        # move to device
        patch1 = patch1.cuda()
        patch2 = patch2.cuda()
        imu = imu.cuda()
        leg = leg.cuda()
        feet = feet.cuda()
        
        # forward pass
        cost = model(patch1.float())
        cost = cost.detach().cpu().numpy().flatten()
        
        for i in range(len(label)):
            if label[i] not in data:
                data[label[i]] = []
            data[label[i]].append(cost[i])
    
    # plot the labels in x axis and the mean cost with std in y axis
    import matplotlib.pyplot as plt
    labels = []
    mean_costs = []
    std_costs = []
    for label in data:
        labels.append(label)
        mean_costs.append(np.mean(data[label]))
        std_costs.append(np.std(data[label]))
    
    # plot vertical bars, labels text in x axis is rotated
    plt.figure()
    plt.bar(labels, mean_costs, yerr=std_costs, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(labels, rotation=45)
    plt.ylabel('Cost')
    plt.title('Costs for different labels')
    # prevent the labels from being cut off
    plt.tight_layout()
        
    # save the plot
    plt.savefig('costs.png')
    
    # draw a boxplot
    plt.figure()
    plt.boxplot(data.values())
    plt.xticks(range(1, len(data)+1), labels, rotation=45)
    plt.ylabel('Cost')
    plt.title('Costs for different labels')
    # prevent the labels from being cut off
    plt.tight_layout()
    
    # save the plot
    plt.savefig('costs_boxplot.png')
    
    
        
        
        
        
    