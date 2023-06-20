import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy

from pytorchtools import EarlyStopping
from custom_reg_dataset import RegisterDataset
from models.SVAE import SVAE
from utils.loss_fn import loss_fn_SVAE

import pytorch_warmup as warmup

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from statistics import mean as mean_calc
import numpy as np
import os
import random

from datetime import datetime
from timeit import default_timer as timer
import time


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cv_fold_maker(train_df, num_of_groups, batch_size, seed = 1, drop_last = False):
    """ 
    FUNCTION DESCRIPTION:
        
    Function for converting validation training data into folds. 
    Converted folds are later on prepared for using in dataloaders. 
    Prepared dataloaders are going to be used in SVAE Training.
    
    
    INPUTS:
    train_df: training data in dataframe format
    num_of_groups: number of groups in the training data. For 5 Folds-CV, it is 5.
    batch_size: batch size to be used in preparation of dataloader.
    seed: random seed value
    drop_last: drop last value for dataloader. 
               For specific batch size values if last batch contains 1 sample, the training gives error.
               When drop_last is True, the last batch in the training is disregarded to avoid error.
               
    OUTPUTS:
    Dataloader_list: {num_of_groups} different cases of dataloader with training folds. 
                    For num_of_groups = 5 ==> [2,3,4,5], [1,3,4,5], [1,2,4,5], [1,2,3,5], [1,2,3,4]
    dataset_val_list: {num_of_groups} different cases of validation. 
                    For num_of_groups = 5 ==> [1],[2],[3],[4],[5]
    num_dims: Number of dimensions(features) in training data.

    

    """
    
    torch.backends.cudnn.deterministic = True # used in order to make training in CUDA deterministic
    
    X_train = train_df.drop(columns = ["persistence_d365","pid","group"]) #dropping non-feature variables.
    num_dims = X_train.shape[1]
    #Defining min-max normalizer
    scaler = MinMaxScaler()
    #fitting the normalizer to the training data 
    scaler.fit(X_train)
    
    #list to contain different cases of dataloader
    Dataloader_list = []
    #list to contain different cases of validation fold
    dataset_val_list = []

    
    for i in range(num_of_groups):
        #validation fold
        CV_X_val_df = train_df[train_df["group"]== i + 1] #choosing only 1 fold to be validation fold
   
        CV_X_val_fold = CV_X_val_df.drop(columns = ["persistence_d365","pid","group"]) #dropping non-feature variables
        CV_y_val_fold = CV_X_val_df["persistence_d365"] # labels(output) of chosen validation fold
        CV_X_val_fold = scaler.transform(CV_X_val_fold) #normalization of validation fold 
        
        #training folds
        CV_X_train_df = train_df[train_df["group"]!= i + 1] #choosing remaining {num_of_groups - 1} folds to be training folds
        
        CV_X_train_folds = CV_X_train_df.drop(columns = ["persistence_d365","pid","group"]) # dropping non-feature variables
        CV_y_train_folds = CV_X_train_df["persistence_d365"] # labels(output) of chosen validation fold
        CV_X_train_folds = scaler.transform(CV_X_train_folds) # normalization of training folds
        
        dataset_train = RegisterDataset(pd.DataFrame(CV_X_train_folds), CV_y_train_folds) # converting training folds into tensors

        dataset_val = RegisterDataset(pd.DataFrame(CV_X_val_fold), CV_y_val_fold) # converting validation fold into tensors
        
        #setting random seeds in order to have shuffling order the same in dataloader
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        


        g = torch.Generator()
        g.manual_seed(seed)


        
        data_loader = DataLoader(
            dataset=dataset_train, batch_size=batch_size, shuffle=False, drop_last = drop_last, worker_init_fn = seed_worker, generator = g)
        
        Dataloader_list.append(data_loader)
        dataset_val_list.append(dataset_val)
        
    return Dataloader_list, dataset_val_list, num_dims

def hyperparameter_tuner(device, dims, num_folds, Dataloader_list, X_val_list, y_val_list, \
                         batch_size_list, w_decay_list, alpha_list, beta_list, lr_list, encoder_layer_list, \
                         classifier_layer_list, latent_size_list, epochs = 500, seed = 1, es_thr = 80, es_patience = 10):
    
    """
    FUNCTION DESCRIPTION:
        
    Function for hyperparameter tuning SVAE. By using dataloaders and validation folds generated by cv_fold_maker function,
    different settings are used and the loss value on validation fold is calculated.
    
    INPUTS:
        
    device: device to be used in training (CUDA or CPU)
    dims: number of dimensions (features) existing in training data
    num_folds: number of folds for cross-validation
    Dataloader_list: The list of training folds for each cv case that are transformed into pytorch dataloader objects
    X_val_list: Validation training datasets for each cv that are moved to device already
    y_val_list: Validation label(output) datasets for each cv that are moved to device already
    batch_size_list: list of candidate batch sizes
    w_decay_list: list of candidate w_decays
    alpha_list: list of candidate alpha values
    beta_list: list of candidate beta values
    lr_list: list of candidate initial learning rate values
    encoder_layer_list: list of candidate encoder and decoder layer architecture
    classifier_layer_list: list of candidate classifier layer architecture
    latent_size_list: list of candidate latent size values
    epochs: limit for maximum epochs. default is 500 epochs
    seed: seed value for initializations. default is 1
    es_thr: threshold for early stopping to wait training to reach to certain epoch before validation loss checks
    es_patience: number of epochs for early stop to wait for improvement. If no improvement is observed in the end, the training is stopped. 
    
    OUTPUTS:
    
    report_df: Report that contains all the setting combinations along with their performance scores, times in different folds.
    loss_list: List of loss logs for each different hyperparameter setting combination and fold
    loss_name_list: names of the loss logs
    
    """


    #Directors for saving files
    current_dir = os.getcwd()
    
    #Folder for recording the loss values
    loss_folder = current_dir + "\loss_values\\"
    if not os.path.exists(loss_folder):
        os.mkdir(loss_folder)
        print("loss_values directory is created under " + current_dir)
    
    #Folder for recording reports of hyperparameter tunings
    report_folder = current_dir + "\hyperparameter_reports\\"
    if not os.path.exists(report_folder):
        os.mkdir(report_folder)
        print("hyperparameter_reports directory is created under " + current_dir)

    main_train_start_date = datetime.now()
    print("Training start date is:", main_train_start_date)
    #Datetime that will be used on naming the training report
    main_train_time_string = main_train_start_date.strftime("%d-%m-%Y_%H-%M-%S")
    
    
    
    #Folder where the models are saved
    model_folder_path = "./net_weights/SVAE_models_" + main_train_time_string 
    
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)
        print(model_folder_path  + " directory is created")
    
    #folder for saving loss files
    loss_date_folder = loss_folder + main_train_time_string + "_loss_logs\\"
    if not os.path.exists(loss_date_folder):
        os.mkdir(loss_date_folder)
        print(main_train_time_string + "_loss_logs" + " directory is created under " + loss_folder)
        
    

    report_file_name = main_train_time_string + "_SVAE_hyperparameter_tuning_report" 
    loss_list = []
    loss_name_list = []
    count = 0
    main_train_start = timer()
    #Columns for information that will be recorded during training.
    report_df = pd.DataFrame(columns = ["Model_number","Seed", "Batch_size", "Encoder_num_neurons", "Encoder_num_hidden_layers", \
                                        "Clf_num_neurons","Clf_num_hidden_layers", "Latent_size", "Alpha", "Beta", \
                                        "Init_lr", "W_decay", \
                                        "CV1_val_first_loss","CV2_val_first_loss","CV3_val_first_loss","CV4_val_first_loss", "CV5_val_first_loss",\
                                        "CV1_val_last_loss","CV2_val_last_loss","CV3_val_last_loss","CV4_val_last_loss", "CV5_val_last_loss",\
                                        "CV1_val_best_loss","CV2_val_best_loss","CV3_val_best_loss","CV4_val_best_loss", "CV5_val_best_loss",\
                                        "CV1_best_loss_epoch","CV2_best_loss_epoch","CV3_best_loss_epoch","CV4_best_loss_epoch","CV5_best_loss_epoch",\
                                        "CV1_best_auroc_epoch","CV2_best_auroc_epoch","CV3_best_auroc_epoch","CV4_best_auroc_epoch","CV5_best_auroc_epoch",\
                                        "CV1_train_time", "CV2_train_time","CV3_train_time","CV4_train_time","CV5_train_time",\
                                        "CV1_val_acc", "CV2_val_acc", "CV3_val_acc", "CV4_val_acc", "CV5_val_acc",\
                                        "CV1_val_auroc", "CV2_val_auroc", "CV3_val_auroc", "CV4_val_auroc", "CV5_val_auroc",\
                                        "CV_avg_val_acc", "CV_avg_val_auroc"])


    #for loops for grid search of hyperparameters
    for batch_size in batch_size_list:
        for w_decay in w_decay_list:
            for alpha in alpha_list:
                for beta in beta_list:
                    for lr in lr_list:
                        for encoder_layers in encoder_layer_list:
                            decoder_layers = encoder_layers[::-1] # decoder is mirrored version of encoder
                            for classifier_layers in classifier_layer_list:
                                for latent_size in latent_size_list:
                                    count += 1
                                    cv_val_first_loss_list = []
                                    cv_val_last_loss_list = []
                                    cv_val_best_loss_list = []
                                    cv_best_loss_epoch_list = []
                                    cv_best_auroc_epoch_list = []
                                    cv_train_time_list = []
                                    cv_val_acc_list = []
                                    cv_val_auroc_list = []
                                    for fold in range(num_folds):
                                        """START OF A FOLD TRAINING"""
                                        cv_start_date = datetime.now()
                                        cv_time_string = cv_start_date.strftime("%d-%m-%Y_%H-%M-%S")

                                        cv_train_start = timer()
                                        loss_file_name_short = "date_" + cv_time_string + "_model_" + str(count) + "_cv_fold_" + str(fold + 1)
                                        
                                        
                                        
                                        loss_file_name = "date_" + cv_time_string + "_dim_" + str(dims) +"_lr_" +str(lr) + "_encoder_layers_" \
                                        + str(encoder_layers[-1]) + "_" + str(len(encoder_layers)-1) + "_clf_layers_" + str(classifier_layers[0]) + "_" + str(len(classifier_layers)-1)\
                                        + "_latent_size_" + str(latent_size) + "_alpha_" + str(alpha) + "_beta_" + str(beta) + "_batch_" + str(batch_size) \
                                        +"_weight_dec_" + str(w_decay)  +"_es_thr_" + str(es_thr) + "_cv_fold_" + str(fold + 1)
                                        
                                        loss_name_list.append(loss_file_name)

                                        loss_df = pd.DataFrame(columns = ["Epoch", "Total_Training_Loss", "Train_Recon_Loss", "Train_KLDiv_Loss_w_Beta", \
                                          "Train_CLF_Loss_w_Alpha","Total_Test_Loss", "Test_Recon_Loss", "Test_KLDiv_Loss_w_Beta", \
                                                                        "Test_CLF_Loss_w_Alpha","Test_AUROC" ,"learning_rate"])
                                        
                                        #new svae instance is created
                                        svae = SVAE(
                                                device=device,
                                                dim_x_h=dims,
                                                encoder_layer_sizes=encoder_layers,
                                                latent_size=latent_size,
                                                decoder_layer_sizes=decoder_layers,
                                                classifier_layer_sizes=classifier_layers, seed= seed).to(device)
                                        
                                        #new adam optimizer with specified learning rate and weight decay for regularization
                                        optimizer = torch.optim.Adam(svae.parameters(), lr=lr, weight_decay=w_decay)
                                        #learning rate decay scheduler
                                        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                                        #warmup scheduler for introducing warmup on learning rate.
                                        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

                                        # initialize the early_stopping object
                                        early_stopping = EarlyStopping(patience=es_patience, delta = 0.01, verbose=True, es_thr = es_thr)
                                        
                                        torch.manual_seed(seed)
                                        torch.cuda.manual_seed(seed)
                                        for epoch in range(epochs):
                                            "START OF EPOCH"
                                            for iteration, (x, y, idx) in enumerate(Dataloader_list[fold]):
                                                "START OF MINI-BATCH"
                                                x, y = x.to(device), y.to(device)
                                                
                                                recon_x, mean, log_var, z, pred_score = svae(x)
                                                #calculating total training loss, reconstruction loss, similarity loss, and classification loss
                                                loss_train, train_BCE, train_KLD, train_CLF = loss_fn_SVAE(recon_x, x, dims, mean, log_var, 
                                                                                                           pred_score, y.unsqueeze(1).float(), beta, alpha)

                                                optimizer.zero_grad()
                                                loss_train.backward()
                                                optimizer.step()
                                                
                                                #checks if warm-up step is below 2000 in order to update learning rate with warmup scheduler
                                                if warmup_scheduler.last_step < 2000:
                                                    with warmup_scheduler.dampening():
                                                        pass
                                                "END OF MINI-BATCH"
                                            #At the end of the epoch, checking if warmup scheduler reached the initial learning rate. Also checks if decayed lr is above 10*-7.
                                            #if satisfied, scheduler decays the learning rate
                                            if optimizer.param_groups[0]["lr"] > 0.0000001 and warmup_scheduler.last_step >= 2000:
                                                lr_scheduler.step()
                                                
                                            #deepcopying the svae model in the end of epoch in order to evaluate performance on all training folds
                                            train_svae = copy.deepcopy(svae)
                                            
                                            #deepcopying the svae model in the end of epoch in order to evaluate performance on validation fold
                                            val_svae = copy.deepcopy(svae)
                                            
                                            #sets train_svae to evaluation mode in order to not update batch normalization layers etc.
                                            train_svae.eval()
                                            
                                            #sets val_svae to evaluation mode in order to not update batch normalization layers etc.
                                            val_svae.eval()
                                            
                                            #no gradient is calculated here
                                            with torch.no_grad():
                                                #getting all training folds data and labels
                                                x_train_folds, y_train_folds = Dataloader_list[fold].dataset.x.to(device), Dataloader_list[fold].dataset.y.to(device)
                                                
                                                #svae used on all training folds
                                                train_folds_recon_x, train_folds_mean, train_folds_log_var, train_folds_z, train_folds_pred_score = train_svae(x_train_folds)
                                                #svae used on validation fold
                                                val_recon_x, val_mean, val_log_var, val_z, val_pred_score = val_svae(X_val_list[fold])
                                                
                                                #auroc score on validation fold is calculated
                                                val_pred_prob = torch.sigmoid(val_pred_score)
                                                val_auroc = roc_auc_score(y_val_list[fold].cpu(), val_pred_prob.detach().cpu())
                                                
                                                
                                                
                                                #training loss is calculated
                                                loss_train_folds, train_folds_BCE, train_folds_KLD, train_folds_CLF = loss_fn_SVAE(train_folds_recon_x,
                                                                                                                                   x_train_folds, dims, train_folds_mean, train_folds_log_var, 
                                                train_folds_pred_score, y_train_folds.unsqueeze(1).float(), beta, alpha)                                                
                                                
                                                
                                                #validation loss is calculated
                                                loss_val, val_BCE, val_KLD, val_CLF = loss_fn_SVAE(val_recon_x, X_val_list[fold], dims, val_mean, val_log_var, 
                                                val_pred_score, y_val_list[fold].unsqueeze(1).float(), beta, alpha)
                                                
                                            #sets train_svae back to training mode
                                            train_svae.train()
                                            #sets val_svae back to training mode
                                            val_svae.train()
                                            #collecting all the loss values and learning rate value in a list to save in loss files.
                                            
                                            #loss for saving with whole training dataset
                                            "USED FOR ALL TRAINING FOLDS TOGETHER"
                                            epoch_loss = [epoch+1, loss_train_folds.item(),train_folds_BCE.item(), train_folds_KLD.item(), train_folds_CLF.item(),\
                                            loss_val.item(), val_BCE.item(), val_KLD.item(), val_CLF.item(), val_auroc, optimizer.param_groups[0]["lr"] ]
                                                

                                                
                                            #saving the loss value calculated at the first epoch
                                            if epoch == 0:
                                                cv_val_first_loss_list.append(loss_val.item())


                                            loss_df.loc[len(loss_df)] = epoch_loss
                                            #checking validation loss and updates the best svae model based on the auroc score it achieved
                                            early_stopping(loss_val.item(),val_auroc, svae)

                                            #if early stop is done, best svae model that has been observed by early stop so far is saved.
                                            if early_stopping.early_stop:
                                                best_svae = copy.deepcopy(early_stopping.best_model)
                                                best_loss_epoch = early_stopping.best_loss_epoch
                                                best_auroc_epoch = early_stopping.best_auroc_epoch
                                                break
                                            """END OF EPOCH"""
                                        try:
                                            loss_df.to_excel(os.path.join(loss_date_folder, loss_file_name_short + ".xlsx"), index = False)
                                        except:
                                            pass
                                        
                                        loss_list.append(loss_df)
                                        
                                        #saving the last validation loss calculated before the training ended
                                        cv_val_last_loss_list.append(loss_val.item())
    
                                        #saving the best validation loss found during the training
                                        cv_val_best_loss_list.append(-early_stopping.best_loss_value)
                                        
                                        #saving the epoch when the best loss value was achieved
                                        cv_best_loss_epoch_list.append(best_loss_epoch)
                                        
                                        #saving the epoch when the best auroc score was achieved
                                        cv_best_auroc_epoch_list.append(best_auroc_epoch)


                                        cv_train_end = timer()

                                        cv_train_time_list.append(time.strftime('%H:%M:%S', time.gmtime(cv_train_end - cv_train_start)))
                                        
                                        #model settings are saved
                                        model_time = datetime.now()
                                        model_dt = model_time.strftime("%d-%m-%Y_%H-%M-%S")
                                        model_path =  os.path.join(model_folder_path, model_dt + "_model_" +str(count) + "_fold_" +str(fold+1)+ ".pth")
                                        torch.save(best_svae.state_dict(), model_path)
                                        
                                        #setting the best svae model to evaluation mode for calculating scores on validation fold
                                        best_svae.eval()
                                        with torch.no_grad():
                                            _, _, _, _, val_pred_score_last = best_svae(X_val_list[fold])
                                            val_pred_prob_last = torch.sigmoid(val_pred_score_last)
                                            threshold = torch.tensor([0.5])
                                            val_pred_labels = (val_pred_prob.detach().cpu()>threshold).float()*1
                                            

                                            
                                        #accuracy on validation fold is calculated
                                        cv_val_acc_list.append(accuracy_score(y_val_list[fold].cpu(), val_pred_labels))
                                        
                                        #auroc calculated on probability with sigmoid
                                        cv_val_auroc_list.append(roc_auc_score(y_val_list[fold].cpu(), val_pred_prob_last.detach().cpu()))
                                    """END OF A FOLD TRAINING"""    
                                        

                                        
                                        


                                    #creating a list of all metrics calculated in this training section
                                    tuning_result = [count, seed, batch_size,encoder_layers[-1], len(encoder_layers) - 1, \
                                                     classifier_layers[0], len(classifier_layers) - 1, latent_size, alpha, beta,\
                                                     lr, w_decay, \
                                                     cv_val_first_loss_list[0],cv_val_first_loss_list[1], cv_val_first_loss_list[2],\
                                                     cv_val_first_loss_list[3], cv_val_first_loss_list[4], \
                                                     cv_val_last_loss_list[0],cv_val_last_loss_list[1], cv_val_last_loss_list[2],\
                                                     cv_val_last_loss_list[3], cv_val_last_loss_list[4], \
                                                     cv_val_best_loss_list[0],cv_val_best_loss_list[1], cv_val_best_loss_list[2],\
                                                     cv_val_best_loss_list[3], cv_val_best_loss_list[4], \
                                                     cv_best_loss_epoch_list[0],cv_best_loss_epoch_list[1], cv_best_loss_epoch_list[2],\
                                                     cv_best_loss_epoch_list[3], cv_best_loss_epoch_list[4], \
                                                     cv_best_auroc_epoch_list[0],cv_best_auroc_epoch_list[1], cv_best_auroc_epoch_list[2],\
                                                     cv_best_auroc_epoch_list[3], cv_best_auroc_epoch_list[4], \
                                                     cv_train_time_list[0],cv_train_time_list[1], cv_train_time_list[2],\
                                                     cv_train_time_list[3], cv_train_time_list[4], \
                                                     cv_val_acc_list[0],cv_val_acc_list[1], cv_val_acc_list[2],\
                                                     cv_val_acc_list[3], cv_val_acc_list[4], \
                                                     cv_val_auroc_list[0], cv_val_auroc_list[1], cv_val_auroc_list[2],\
                                                     cv_val_auroc_list[3], cv_val_auroc_list[4], \
                                                     mean_calc(cv_val_acc_list), mean_calc(cv_val_auroc_list)]
                                        
                                    
                                    #adding the recorded metrics to report
                                    report_df.loc[len(report_df)] = tuning_result
                                    #updating the file with each finished training
                                    report_df.to_csv(os.path.join(report_folder, report_file_name + ".txt"), sep = "\t", index = False)
                                    if count % 5 == 1: 
                                        print(str(count) + " settings have been checked and saved to report file")


    
    report_df.to_csv(os.path.join(report_folder, report_file_name + ".txt"), sep = "\t", index = False)

    main_train_end_date = datetime.now()
    print("Training end date is:", main_train_end_date)

    main_train_end = timer() 
    main_train_time = main_train_end - main_train_start

    print("Total Training Time is:", time.strftime('%H:%M:%S', time.gmtime(main_train_time)))
    return report_df, loss_list, loss_name_list


def model_test(device, train_df, test_df, batch_size, w_decay, alpha, beta, lr_init, encoder_layers, classifier_layers,latent_size, 
               epochs = 500, seed = 1, es_active = True, es_thr = 80, es_patience = 10, datestring = "no_date_specified", shuffling = False):
    """
    FUNCTION DESCRIPTION:
        
    Function for testing a SVAE model with specific hyperparameter settings on provided test data. 
    In contrast to hyperparameter_tuner function, no cross validation is done in this function. Only 1 training is done with whole provided training data.
    Then the performance metrics are calculated on provided test data.

    
    INPUTS:
    device: device to be used in training (CUDA or CPU)
    train_df: training data in dataframe
    test_df: test data in dataframe
    batch_size: batch size to be used in training
    w_decay: Weight decay to be used in regularization through adam optimizer
    alpha: multiplier for classification loss
    beta: multiplier for similarity loss
    lr_init: inital learning rate
    encoder_layers: encoder layers with specified number of layers and neurons
    classifier_layers: classifier layers with specified number of layers and neurons
    lr_list: list of candidate initial learning rate values
    encoder_layer_list: list of candidate encoder and decoder layer architecture
    classifier_layer_list: list of candidate classifier layer architecture
    latent_size: number of dimensions in latent space
    epochs: limit for maximum epochs. default is 500 epochs
    seed: seed value for initializations. default is 1
    es_active: Boolean trigger value for using early stop functionality. Default is True meaning early stop is active.
    es_thr: threshold for early stopping to wait training to reach to certain epoch before validation loss checks
    es_patience: number of epochs for early stop to wait for improvement. If no improvement is observed in the end, the training is stopped. 
    datestring: date to be used in naming of folders
    shuffling: shuffles the data or not
    
    OUTPUTS:
        
    best_auroc_epoch:The epoch number that the model with the highest auroc score was trained
    best_loss_epoch:The epoch number that the model with the lowest validation loss was trained
    acc_score: Accuracy score calculated on test data
    auroc_score: AUROC score calculated on test data
    loss_df: Dataframe containing different types of loss values calculated on each epoch
    y_test: true labels of test data
    eval_pred_labels:predicted labels of test data
    eval_pred_labels_score: predicted label scores of test data
    
    
    """
    # decoder is mirrored version of encoder
    decoder_layers = encoder_layers[::-1]

    
    #non-feature columns are dropped. If "group" column doesn't exist, it is skipped
    try:
        X_train = train_df.drop(columns = ["persistence_d365","pid","group"])
    except:
        X_train = train_df.drop(columns = ["persistence_d365","pid"])
    #assigning labels of training data
    y_train = train_df["persistence_d365"]
    
    #number of dimensions(features) in data
    dims = X_train.shape[1]
    
    #non-feature columns are dropped. If "group" column doesn't exist, it is skipped
    try:
        X_test = test_df.drop(columns = ["persistence_d365","pid","group"])
    except:
        X_test = test_df.drop(columns = ["persistence_d365","pid"])
        
    #assigning labels of test data
    y_test = test_df["persistence_d365"]
    
    #defining min-max normalizer
    scaler = MinMaxScaler()
    
    #min-max normalizer is fit to training data and both training and test data is normalized based on fitted normalizer
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #moving normalized data to tensors
    dataset_train = RegisterDataset(pd.DataFrame(X_train_scaled), y_train)
    dataset_test = RegisterDataset(pd.DataFrame(X_test_scaled), y_test)
    
    #setting random seed for controlling shuffle in dataloader
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


    
    data_loader = DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=shuffling, worker_init_fn = seed_worker, generator = g)
#    data_loader = DataLoader(
#            dataset=dataset_train, batch_size=batch_size, shuffle=True)

    #moving test data to device (CUDA for GPU Calculation)
    X_test = dataset_test.x
    X_test = X_test.to(device)
    
    y_test = dataset_test.y
    y_test = y_test.to(device)
    

    
    #Directors for saving files
    current_dir = os.getcwd()
    
    #Folder for recording the loss values
    loss_folder = current_dir + "\loss_values\\"
    if not os.path.exists(loss_folder):
        os.mkdir(loss_folder)
        print("loss_values directory is created under " + current_dir)
        
        #Folder where the models are saved
    model_folder_path = "./net_weights/SVAE_models_" + datestring + "/"
    
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)
        print(model_folder_path  + " directory is created")
        
        #folder for saving loss files
    loss_date_folder = loss_folder + datestring + "_loss_logs\\"
    if not os.path.exists(loss_date_folder):
        os.mkdir(loss_date_folder)
        print(datestring + "_loss_logs" + " directory is created under " + loss_folder)
        
       

    #Creating dataframe for storing different loss types for each epoch. Test AUROC and Learning rate for each epoch is recorded too. 
    loss_df = pd.DataFrame(columns = ["Epoch", "Total_Training_Loss", "Train_Recon_Loss", "Train_KLDiv_Loss_w_Beta", \
      "Train_CLF_Loss_w_Alpha","Total_Test_Loss", "Test_Recon_Loss", "Test_KLDiv_Loss_w_Beta", \
                                    "Test_CLF_Loss_w_Alpha","Test_AUROC", "learning_rate"])
        
    loss_file_name_short = "date_" + datestring + "_seed_" + str(seed)
    
    svae = SVAE(
            device=device,
            dim_x_h=dims,
            encoder_layer_sizes=encoder_layers,
            latent_size=latent_size,
            decoder_layer_sizes=decoder_layers,
            classifier_layer_sizes=classifier_layers, seed= seed).to(device)

    optimizer = torch.optim.Adam(svae.parameters(), lr=lr_init, weight_decay=w_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # initialize the early_stopping object if early stop is active
    if es_active:
        early_stopping = EarlyStopping(patience= es_patience, delta = 0.01, verbose=True, es_thr = es_thr)
                                        
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for epoch in range(epochs):
        for iteration, (x, y, idx) in enumerate(data_loader):
            #training data and labels cominng from the batch are moved to device 
            x, y = x.to(device), y.to(device)
            #data is fed to svae and los values are calculated on training data
            recon_x, mean, log_var, z, pred_score = svae(x)
            loss_train, train_BCE, train_KLD, train_CLF = loss_fn_SVAE(recon_x, x, dims, mean, log_var, pred_score, y.unsqueeze(1).float(), beta, alpha)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            #checks if warm-up step is below 2000 in order to update learning rate with warmup scheduler
            if warmup_scheduler.last_step < 2000:
                with warmup_scheduler.dampening():
                    pass
                
        #At the end of the epoch, checking if warmup scheduler reached the initial learning rate. Also checks if decayed lr is above 10*-7.
        #if satisfied, scheduler decays the learning rate
        if optimizer.param_groups[0]["lr"] > 0.0000001 and warmup_scheduler.last_step >= 2000:
            lr_scheduler.step()
            
          
        #deepcopying the svae model in the end of epoch in order to evaluate performance on training data
        train_svae = copy.deepcopy(svae)
        
        #deepcopying the svae model in the end of epoch in order to evaluate performance on test data
        test_svae = copy.deepcopy(svae)
        
        #sets train_svae to evaluation mode in order to not update batch normalization layers etc.
        train_svae.eval()
        
        #sets test_svae to evaluation mode in order to not update batch normalization layers etc.
        test_svae.eval()
        
        #no gradient is calculated here
        with torch.no_grad():
            #getting all training data and labels
            X_train, y_train = data_loader.dataset.x.to(device), data_loader.dataset.y.to(device)
            
            #svae used on all training data
            train_all_recon_x, train_all_mean, train_all_log_var, train_all_z, train_all_pred_score = train_svae(X_train)
            #svae used on test data
            test_recon_x, test_mean, test_log_var, test_z, test_pred_score = test_svae(X_test)
            
            #auroc score on validation fold is calculated
            test_pred_prob = torch.sigmoid(test_pred_score)
            test_auroc = roc_auc_score(y_test.cpu(), test_pred_prob.detach().cpu())
            
            
            
            #training loss is calculated
            loss_train_all, train_all_BCE, train_all_KLD, train_all_CLF = loss_fn_SVAE(train_all_recon_x,
                                                                                               X_train, dims, train_all_mean, train_all_log_var, 
            train_all_pred_score, y_train.unsqueeze(1).float(), beta, alpha)                                                
            
            
            #test loss is calculated
            loss_test, test_BCE, test_KLD, test_CLF = loss_fn_SVAE(test_recon_x, X_test, dims, test_mean, test_log_var, 
            test_pred_score, y_test.unsqueeze(1).float(), beta, alpha)
            
        #sets train_svae back to training mode
        train_svae.train()
        #sets test_svae back to training mode
        test_svae.train()

        
        #collecting all the loss values and learning rate value in a list to save in loss files.
        epoch_loss = [epoch+1, loss_train_all.item(),train_all_BCE.item(), train_all_KLD.item(), train_all_CLF.item(),\
        loss_test.item(), test_BCE.item(), test_KLD.item(), test_CLF.item(),test_auroc, optimizer.param_groups[0]["lr"] ]



        #the loss list is added to loss dataframe
        loss_df.loc[len(loss_df)] = epoch_loss
        
        #if early stop is active, the latest loss value is checked and the training is stopped if conditions are satisfied
        if es_active:
            early_stopping(loss_test.item(),test_auroc, svae)
            #if early stop is satisfied, the best model along with its corresponding epoch is saved and the training is stopped
            if early_stopping.early_stop:
                best_svae = copy.deepcopy(early_stopping.best_model)
                best_loss_epoch = early_stopping.best_loss_epoch
                best_auroc_epoch = early_stopping.best_auroc_epoch
                break


    try:
        loss_df.to_excel(os.path.join(loss_date_folder, loss_file_name_short + ".xlsx"), index = False)
    except:
        pass

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    
    
    #model settings are saved
    
    PATH = model_folder_path + 'register_net_svae_'+ dt_string +  "_seed_" + str(seed)   + '.pth'

    if es_active:
        
        #Saving best svae model from early stop
        torch.save(best_svae.state_dict(), PATH)
        
        best_svae.eval()
        with torch.no_grad():
            
            _, _, _, eval_z, eval_pred_score = best_svae(X_test)
            eval_pred_prob = torch.sigmoid(eval_pred_score)
            threshold = torch.tensor([0.5])
            # if probability is <0.5, then label is 0, otherwise label is 1
            eval_pred_labels = (eval_pred_prob.detach().cpu()>threshold).float()*1
            
    #otherwise, the model in the end of the training after specified number of epochs is used for performance metric calculation on test data
    else:
        
        #Saving the svae model in the end of the training without early stop
        torch.save(svae.state_dict(), PATH)
        
        svae.eval()
        with torch.no_grad():
            _, _, _, eval_z, eval_pred_score = svae(X_test)
            eval_pred_prob = torch.sigmoid(eval_pred_score)
            threshold = torch.tensor([0.5])
            # if probability is <0.5, then label is 0, otherwise label is 1
            eval_pred_labels = (eval_pred_prob.detach().cpu()>threshold).float()*1
            

    
    #accuracy on validation fold is calculated
    acc_score = accuracy_score(y_test.cpu(), eval_pred_labels)
                                    
    #auroc calculated on probability with sigmoid
    auroc_score = roc_auc_score(y_test.cpu(), eval_pred_prob.detach().cpu())
    
    print("Accuracy Score on the test data is:", acc_score)
    print("ROC-AUC Score on the test data is:", auroc_score)
    

    
    return  best_auroc_epoch, best_loss_epoch, acc_score, auroc_score, X_test, y_test, eval_z, eval_pred_labels, eval_pred_prob



def latent_corr_calc(data_tensor, data_columns, latent_tensor, top_amount = None):
    """FUNCTION DESCRIPTION
    Function for calculating correlation between variables and latent dimensions"""
    data_tensor_cpu = data_tensor.detach().cpu()
    data_df = pd.DataFrame(data_tensor_cpu.numpy(), columns = data_columns)
    
    latent_tensor_cpu = latent_tensor.detach().cpu()

    latent_f_name_list = []
    for f_num in range(latent_tensor_cpu.shape[1]):
        latent_f_name_list.append("latent_feature_" + str(f_num+1))

    latent_df = pd.DataFrame(latent_tensor_cpu.numpy(), columns = latent_f_name_list)
    
    
    comnined_df = pd.concat([data_df, latent_df], axis=1)
    comnined_df_corr = comnined_df.corr()
    
    comnined_df_corr_cleaned = comnined_df_corr[latent_f_name_list].drop(index = latent_f_name_list)
    
    comnined_df_corr_cleaned_abs = comnined_df_corr_cleaned.abs()
    comnined_df_corr_unstack = comnined_df_corr_cleaned_abs.unstack()
    comnined_df_corr_sorted = comnined_df_corr_unstack.sort_values(kind="quicksort",ascending = False )
    
    latent_f_top_vars_list = []
    
    for latent_feature in latent_f_name_list:
        top_condition = comnined_df_corr_sorted.index.get_level_values(0) == latent_feature
        if top_amount == None:
            latent_f_top_vars = comnined_df_corr_sorted[top_condition]
        else:
            latent_f_top_vars = comnined_df_corr_sorted[top_condition][:top_amount]
        latent_f_top_vars_list.append(latent_f_top_vars)
        
    return latent_f_top_vars_list