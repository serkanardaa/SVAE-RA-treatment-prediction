#The following code is an adaptation of the source code presented in https://github.com/tianchenji/Multimodal-SVAE

import torch
import torch.nn as nn

from .blocks.MLP import MLP
from .blocks.Encoder import Encoder
from .blocks.Decoder import Decoder

class SVAE(nn.Module):

    def __init__(self, device,  dim_x_h, encoder_layer_sizes, latent_size,
                 decoder_layer_sizes, classifier_layer_sizes, seed = 1):

        super().__init__()

        self.seed = seed
        self.dim_x_h = dim_x_h

        self.device = device

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, self.seed)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, self.seed)

        latent_para_size = latent_size * 2
        classifier_input_size = latent_para_size 

        classifier_layer_sizes = [classifier_input_size] + classifier_layer_sizes

        self.classifier = MLP(classifier_layer_sizes, self.seed)

    def forward(self, x_h):

        # flatten the image-like high-dimensional inputs x_h
        if x_h.dim() > 2:
            x_h = x_h.view(-1, self.dim_x_h)

        batch_size = x_h.size(0)

        means, log_var = self.encoder(x_h)

        std = torch.exp(0.5 * log_var)
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        self.seed += 1
        z = eps * std + means

        classifier_inputs = torch.cat((means, log_var), dim=-1)

        pred_labels_score = self.classifier(classifier_inputs)

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z, pred_labels_score

