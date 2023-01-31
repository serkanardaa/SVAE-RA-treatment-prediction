import torch.nn as nn
import torch

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, batch_norm = True, seed = 1):

        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size
        
        #Checks if batch normalization is active. Default:Active
        if batch_norm:

            for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
                #Providing Random Seed before layer initialization
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                layer = nn.Linear(in_size, out_size)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
                
                self.MLP.add_module(
                    name="L{:d}".format(i), module=layer)
                if i+1 < len(layer_sizes):
                    self.MLP.add_module(name="BN{:d}".format(i),module = nn.BatchNorm1d(out_size))
                    self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                else:
                    self.MLP.add_module(name="BN{:d}".format(i),module = nn.BatchNorm1d(out_size))
                    self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
                #incrementing provided seed in order not to have exact the same weights in upcoming layers
                seed += 1
        
        else:
        
            for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                layer = nn.Linear(in_size, out_size)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity = "relu")
                
                self.MLP.add_module(
                    name="L{:d}".format(i), module=layer)
                if i+1 < len(layer_sizes):
                    self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                else:
                    self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
                seed += 1

    def forward(self, z):

        x = self.MLP(z)

        return x