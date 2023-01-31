import torch




def loss_fn_SVAE(recon_x, x, num_points, mean, log_var, pred_labels_score, y, beta = 1, alpha = 1):


    #reconstruction loss
    BCE = torch.nn.functional.mse_loss(recon_x.view(-1, num_points), x.view(-1, num_points), reduction='sum')
    
    #KL Divergence Loss (Similarity Loss)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    #Classification loss in cross entropy
    CLF = torch.nn.functional.cross_entropy(pred_labels_score, y, reduction='sum')
    
    return (BCE + beta * KLD + alpha * CLF) / x.size(0), BCE/ x.size(0), (beta * KLD)/ x.size(0), (alpha * CLF)/ x.size(0)

def loss_fn_generative(recon_x, x, num_points, mean, log_var):

    BCE = torch.nn.functional.mse_loss(
        recon_x.view(-1, num_points), x.view(-1, num_points), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

def loss_fn_discriminative(pred_labels_score, y):

    CLF = torch.nn.functional.cross_entropy(pred_labels_score, y, reduction='sum')

    return CLF / y.size(0)