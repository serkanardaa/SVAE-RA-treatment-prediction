#The following code is an adaptation of the source code presented in https://github.com/tianchenji/Multimodal-SVAE

import torch



def loss_fn_SVAE(recon_x, x, num_points, mean, log_var, pred_labels_score, y, beta = 1, alpha = 1):


    #reconstruction loss
    BCE = torch.nn.functional.mse_loss(recon_x.view(-1, num_points), x.view(-1, num_points), reduction='sum')
    
    #KL Divergence Loss (Similarity Loss)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    #Classification loss in cross entropy
    CLF = torch.nn.functional.binary_cross_entropy_with_logits(pred_labels_score, y, reduction='sum')
    
    return (BCE + beta * KLD + alpha * CLF) / x.size(0), BCE/ x.size(0), (beta * KLD)/ x.size(0), (alpha * CLF)/ x.size(0)

