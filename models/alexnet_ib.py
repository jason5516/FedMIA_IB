from .alexnet import *
from torch import nn 

__all__ = ['alexnet_IB']

class AlexNet_IB(nn.Module):
    def __init__(self, num_classes=10, dim=256, latent_size = 256):
        super().__init__()

        self.dim = dim
        self.beta = 1.0
        
        self.feature_extractor = alexnet(num_classes=num_classes)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        self.fc_mu  = nn.Linear(dim * 4 * 4, latent_size) 
        self.fc_std = nn.Linear(dim * 4 * 4, latent_size)

        # for classfication
        self.decoder = nn.Linear(latent_size, num_classes)

    
    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.feature_extractor(x)
        # flatten
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)
    
    def decode(self, z):

        return self.decoder(z)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def loss_function(self, mu, std):
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return KL 
    
    def get_features(self, x):
        # x = self.encoder(x)

        mu, std = self.encode(x)
        z = self.reparameterise(mu, std)

        return z
    
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        mu, std = self.encode(x)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)

        # loss = self.loss_function(output, label_ids, mu, std, self.beta)
        KL_loss = self.loss_function(mu, std)

        return output, KL_loss

def alexnet_IB(num_classes):
    return AlexNet_IB(num_classes)
