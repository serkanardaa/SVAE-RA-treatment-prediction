import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, batch_norm = True, seed = 1):

        super().__init__()

        self.MLP = nn.Sequential()
        
        #Checks if batch normalization is active. Default:Active
        if batch_norm:
            
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                #Providing Random Seed before layer initialization
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                layer = nn.Linear(in_size, out_size)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
                
                self.MLP.add_module(
                    name="L{:d}".format(i), module=layer)
                self.MLP.add_module(name="BN{:d}".format(i),module = nn.BatchNorm1d(out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                #incrementing provided seed in order not to have exact the same weights in upcoming layers
                seed += 1
        else:
            
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                layer = nn.Linear(in_size, out_size)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
                
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())   
                seed += 1
                
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        nn.init.kaiming_uniform_(self.linear_means.weight, nonlinearity = "relu")
        
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 1)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        nn.init.kaiming_uniform_(self.linear_log_var.weight, nonlinearity = "relu")

    def forward(self, x):

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars