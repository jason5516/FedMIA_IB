from .resnet_cifar import *
from .ib_layers import InformationBottleneck
from torch import nn 

__all__ = ['ResNet18_IB']

class Resnet18_IB_my(nn.Module):
    def __init__(self, num_classes=10, dim=512, r_dim = 64, latent_size = 256):
        super().__init__()
        self.r_dim = r_dim
        self.dim = dim
        self.beta = 1.0
        
        self.feature_extractor = ResNet18(num_classes=num_classes)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        self.e = nn.Sequential(nn.Flatten(), nn.Linear(dim * 16, r_dim * 2) )
        self.dil = nn.Sequential(
                nn.Linear(r_dim, latent_size),
                nn.LeakyReLU(0.2, inplace=True),
        )


        # for classfication
        self.decoder = nn.Linear(latent_size, num_classes)

    
    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.feature_extractor(x)
        return x

    
    def decode(self, z):

        return self.decoder(z)
    
    def reparameterise(self, z):
        code = self.e(z)
        mu = code[:, :self.r_dim]
        var = F.softplus(code[:, self.r_dim:]) + 1e-5
        scale_tri = torch.diag_embed(var)
        return torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=scale_tri)
    
    def loss_function(self, r_dist):
        M_r = torch.distributions.multivariate_normal.MultivariateNormal(loc = torch.zeros(self.r_dim).to(r_dist.loc.device),
                                                                         scale_tril=torch.eye(self.r_dim, self.r_dim).to(r_dist.loc.device)
                                                                         )
        return torch.distributions.kl.kl_divergence(r_dist, M_r).mean()
    
    def get_features(self, x):
        # x = self.encoder(x)

        z = self.encode(x)
        r_dist = self.reparameterise(z)

        return r_dist
    
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        z = self.encode(x)
        r_dist = self.reparameterise(z)
        z1 = self.dil(r_dist.sample())
        output =  self.decode(z1)

        # loss = self.loss_function(output, label_ids, mu, std, self.beta)
        KL_loss = self.loss_function(r_dist)

        return output, KL_loss

class Resnet18_IB(nn.Module):
    def __init__(self, num_classes=10, dim=512, latent_size = 256):
        super().__init__()

        self.dim = dim
        self.beta = 1.0
        
        self.feature_extractor = ResNet18(num_classes=num_classes)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        # linear IB structure
        self.fc_mu  = nn.Linear(dim * 4 * 4, latent_size) 
        self.fc_std = nn.Linear(dim * 4 * 4, latent_size)

        # convolutional IB structure
        self.conv_mu = nn.Conv2d(dim, latent_size,
                                  kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(dim, latent_size,
                                     kernel_size=3, padding=1)

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
        log_var = 2 * torch.log(std + 1e-8)
        KL = 0.5 * torch.mean(mu.pow(2) + log_var.exp() - log_var - 1)
        # origenal KL loss
        # KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
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

class Resnet18_IB_least(nn.Module):
    def __init__(self, num_classes=10, dim=512, latent_size = 128):
            super().__init__()

            self.dim = dim
            self.beta = 1.0
            
            self.feature_extractor = ResNet18(num_classes=num_classes)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

            # IB structure
            self.ib = InformationBottleneck(dim)

            # for classfication
            self.decoder = nn.Linear(latent_size * 8 * 8, num_classes)
    
    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.feature_extractor(x)
        # flatten
        # x = x.view(x.size(0), -1)
        return x
    
    def decode(self, z):

        return self.decoder(z)
    
    def forward(self, x):

        out = self.encode(x)
        z = self.ib(out)

        z = z.view(z.size(0), -1)
        output =  self.decode(z)

        KL_loss = self.ib.kld

        return output, KL_loss

def ResNet18_IB(num_classes):
    return Resnet18_IB_least(num_classes)

# def ResNet18_IB(num_classes):
#     return Resnet18_CIB(num_classes)
