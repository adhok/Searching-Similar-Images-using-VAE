import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import time
import pandas as pd
import subprocess
from PIL import Image


d = 20
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
        nn.BatchNorm2d(2)
        )
        self.fc1 = nn.Sequential(
            ### Reduce the number of channels to 1 without changing the width and dimensions of the images
            nn.Linear(26*26*2,128),
            
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,64),
            
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,32)
            
        )

        self.encoder = nn.Sequential(
            
            nn.Linear(32, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )
        

        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 32)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,26*26*2),
            nn.ReLU(),
            nn.BatchNorm1d(26*26*2)
            
        )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=3,stride=1,padding=0),
            
            nn.Sigmoid()
        
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            ## Using log variance to ensure that we get a positive std dev
            ## Converting to std dev in the real space
            std = logvar.mul(0.5).exp_()
            ### Create error term which has the same shape as std dev sampled from a N(0,1) distribution
            eps = std.data.new(std.size()).normal_()
            #eps = torch.zeros(std.size())
            ### Add the mean and the std_dev 
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        
        #fc1_output = self.fc1(x.view(-1, 28*28*3))
        conv1_output = self.conv1(x)
        #print(conv1_output.size())
        fc1_output = self.fc1(conv1_output.view(-1,26*26*2))
        
        ### Convert Encoded vector into shape (N,2,d)
        mu_logvar = self.encoder(fc1_output).view(-1, 2, d)
        ### First vector for each image is mean of the latent distribution
        mu = mu_logvar[:, 0, :]
        ### Second vector for each image is log-variance of the latent distribution
        logvar = mu_logvar[:, 1, :]
        ### Create variable Z = mu + error * Std_dev
        z = self.reparameterise(mu, logvar)
        ### Get decoder output
        decoder_output = self.decoder(z)
        
        fc2_output = self.fc2(decoder_output)
        tconv1_output = self.tconv1(fc2_output.view(fc2_output.size(0),2,26,26))
        ## Resize Decoder Output to Pass it to TransposedConv2d layer to recontruct 3 channeled image
        #decoder_output = decoder_output.view(decoder_output.size(0),1,28,28) 
        ## Return Reconstructed Output and mean and log-variance
        return tconv1_output, mu, logvar,z



        


            

            
            

def extract(img):
    """
    Extract a deep feature from an input image
    Args:
        img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

    Returns:
        feature (np.ndarray): deep feature with the shape=(4096, )
    """

    def load_checkpoint(filepath):

        model = VAE()

        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()

        return model


    model_inference = load_checkpoint('./model/checkpoint.pth')
    img = Image.open(img)
    transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])

    img = transform(img)
    img = img.view(1,3,28,28)
    _,mu,_,_ = model_inference(img)
    feature = mu.detach().numpy()  # (1, 4096) -> (4096, )
    return feature  # Normalize

