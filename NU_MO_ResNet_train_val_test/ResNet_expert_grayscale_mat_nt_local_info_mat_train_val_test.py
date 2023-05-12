
import pandas as pd
from data_preparation_grayscale_mat_nt_localized_info_mat_function import data_prep_grayscale_mat_nt_localized_info_mat
import torch
import torch.nn as nn
from train_val_test_plot_functions import create_weighted_sampler, dataloader_prep_with_sampler, dataloader_prep, train_val_ResNet_expert, test_ResNet_expert, train_val_figure_plot_function
from ResNet_architecture_grayscale_mat_nt_localized_info_mat import ResNet_18_pair_grayscale_mat_nt_localized_info_mat



#%%
###### First Part: Data Preparation
# 1.1 Read the data, select the utilized columns, and rename the column name.

# Training data
Original_RNA_Data_train = pd.read_csv('matthews_MFENFE2_training_data.csv')
#Original_RNA_Data_train = Original_RNA_Data_train.head(10)

Original_RNA_Data_train = Original_RNA_Data_train[['name', 'sequence', 'correct_structure', 'RNAfold_structure', 'MCC_RNAfold']]

Original_RNA_Data_train = Original_RNA_Data_train.rename(columns={'MCC_RNAfold' : 'MCC'})


# Validation data
Original_RNA_Data_val = pd.read_csv('matthews_MFENFE2_validation_data.csv')
#Original_RNA_Data_val = Original_RNA_Data_val.head(10)

Original_RNA_Data_val = Original_RNA_Data_val[['name', 'sequence', 'correct_structure', 'RNAfold_structure', 'MCC_RNAfold']]

Original_RNA_Data_val = Original_RNA_Data_val.rename(columns={'MCC_RNAfold' : 'MCC'})


# Testing data
Original_RNA_Data_test = pd.read_csv('matthews_MFENFE2_testing_data.csv')
#Original_RNA_Data_test = Original_RNA_Data_test.head(10)

Original_RNA_Data_test = Original_RNA_Data_test[['name', 'sequence', 'correct_structure', 'RNAfold_structure', 'MCC_RNAfold']]

Original_RNA_Data_test = Original_RNA_Data_test.rename(columns={'MCC_RNAfold' : 'MCC'})


# Rfam testing data
Original_RNA_Data_test_rfam = pd.read_csv('rfam_mcc_summary.csv')
#Original_RNA_Data_test_rfam = Original_RNA_Data_test_rfam.head(10)

Original_RNA_Data_test_rfam = Original_RNA_Data_test_rfam[['name', 'sequence', 'correct_structure', 'RNAfold_structure', 'MCC_RNAfold']]

Original_RNA_Data_test_rfam = Original_RNA_Data_test_rfam.rename(columns={'MCC_RNAfold' : 'MCC'})


#%%
# Training data
train_x_color_mat_np, train_x_nt_localized_info_mat_np, train_y_np = data_prep_grayscale_mat_nt_localized_info_mat(Original_RNA_Data = Original_RNA_Data_train)

train_x_color_mat_4D_torch = torch.from_numpy(train_x_color_mat_np)
train_x_color_mat_4D_torch = train_x_color_mat_4D_torch.type(torch.float)

train_x_nt_localized_info_mat_4D_torch = torch.from_numpy(train_x_nt_localized_info_mat_np)
train_x_nt_localized_info_mat_4D_torch = train_x_nt_localized_info_mat_4D_torch.type(torch.float)

train_y_torch = torch.from_numpy(train_y_np)
train_y_torch = train_y_torch.type(torch.long)

# Validation data
val_x_color_mat_np, val_x_nt_localized_info_mat_np, val_y_np = data_prep_grayscale_mat_nt_localized_info_mat(Original_RNA_Data = Original_RNA_Data_val)

val_x_color_mat_4D_torch = torch.from_numpy(val_x_color_mat_np)
val_x_color_mat_4D_torch = val_x_color_mat_4D_torch.type(torch.float)

val_x_nt_localized_info_mat_4D_torch = torch.from_numpy(val_x_nt_localized_info_mat_np)
val_x_nt_localized_info_mat_4D_torch = val_x_nt_localized_info_mat_4D_torch.type(torch.float)

val_y_torch = torch.from_numpy(val_y_np)
val_y_torch = val_y_torch.type(torch.long)

# Testing data
test_x_color_mat_np, test_x_nt_localized_info_mat_np, test_y_np = data_prep_grayscale_mat_nt_localized_info_mat(Original_RNA_Data = Original_RNA_Data_test)

test_x_color_mat_4D_torch = torch.from_numpy(test_x_color_mat_np)
test_x_color_mat_4D_torch = test_x_color_mat_4D_torch.type(torch.float)

test_x_nt_localized_info_mat_4D_torch = torch.from_numpy(test_x_nt_localized_info_mat_np)
test_x_nt_localized_info_mat_4D_torch = test_x_nt_localized_info_mat_4D_torch.type(torch.float)

test_y_torch = torch.from_numpy(test_y_np)
test_y_torch = test_y_torch.type(torch.long)

# Rfam testing data
rfam_test_x_color_mat_np, rfam_test_x_nt_localized_info_mat_np, rfam_test_y_np = data_prep_grayscale_mat_nt_localized_info_mat(Original_RNA_Data = Original_RNA_Data_test_rfam)

rfam_test_x_color_mat_4D_torch = torch.from_numpy(rfam_test_x_color_mat_np)
rfam_test_x_color_mat_4D_torch = rfam_test_x_color_mat_4D_torch.type(torch.float)

rfam_test_x_nt_localized_info_mat_4D_torch = torch.from_numpy(rfam_test_x_nt_localized_info_mat_np)
rfam_test_x_nt_localized_info_mat_4D_torch = rfam_test_x_nt_localized_info_mat_4D_torch.type(torch.float)

rfam_test_y_torch = torch.from_numpy(rfam_test_y_np)
rfam_test_y_torch = rfam_test_y_torch.type(torch.long)


#%%

sampler_weight_data_loader_utilized = create_weighted_sampler(y_numpy = train_y_np)

device_utilized = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NN_architecture_utilized = ResNet_18_pair_grayscale_mat_nt_localized_info_mat(nt_info_mat_input_num_channels=1, grayscale_mat_input_num_channels=4)

NN_model_utilized = NN_architecture_utilized.to(device_utilized)

# Training Part 2.3: Set hyperparameters, loss function, optimizer, and scheduler.
learning_rate_value_utilized = 0.0001
weight_decay_value_utilized = 0.1
num_epochs_utilized = 100
batch_size_value_utilized = 20
gamma_value_utilized = 0.95

optimizer_utilized = torch.optim.Adam(NN_model_utilized.parameters(), lr=learning_rate_value_utilized, weight_decay=weight_decay_value_utilized)
scheduler_for_optimizer_utilized = torch.optim.lr_scheduler.ExponentialLR(optimizer_utilized, gamma=gamma_value_utilized, last_epoch=-1)

loss_function_utilized = nn.CrossEntropyLoss()

model_name_utilized = 'NUMO_ResNet_18_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'_criteria_1_RNAFold_data.pth'



# Training Part 2.4: Prepare the Dataloader

training_data_utilized = dataloader_prep_with_sampler(x_color_mat_stack_4D_tensor = train_x_color_mat_4D_torch,
                                                      x_nt_localized_info_mat_stack_4D_tensor = train_x_nt_localized_info_mat_4D_torch,
                                                      y_tensor = train_y_torch, 
                                                      batch_size_value = batch_size_value_utilized, 
                                                      sampler_applied = sampler_weight_data_loader_utilized)


val_data_utilized = dataloader_prep(x_color_mat_stack_4D_tensor = val_x_color_mat_4D_torch,
                                    x_nt_localized_info_mat_stack_4D_tensor = val_x_nt_localized_info_mat_4D_torch,
                                    y_tensor = val_y_torch, 
                                    batch_size_value = batch_size_value_utilized)


testing_data_utilized = dataloader_prep(x_color_mat_stack_4D_tensor = test_x_color_mat_4D_torch,
                                        x_nt_localized_info_mat_stack_4D_tensor = test_x_nt_localized_info_mat_4D_torch,
                                        y_tensor = test_y_torch, 
                                        batch_size_value = batch_size_value_utilized)


rfam_testing_data_utilized = dataloader_prep(x_color_mat_stack_4D_tensor = rfam_test_x_color_mat_4D_torch, 
                                             x_nt_localized_info_mat_stack_4D_tensor = rfam_test_x_nt_localized_info_mat_4D_torch, 
                                             y_tensor = rfam_test_y_torch, 
                                             batch_size_value = batch_size_value_utilized)


#%%
train_val_record_obtained, saved_models_metrics_obtained = train_val_ResNet_expert(device = device_utilized, 
                                                                                   NN_model = NN_model_utilized, 
                                                                                   num_epochs = num_epochs_utilized, 
                                                                                   optimizer = optimizer_utilized, 
                                                                                   scheduler_for_optimizer = scheduler_for_optimizer_utilized, 
                                                                                   loss_function = loss_function_utilized, 
                                                                                   model_name = model_name_utilized, 
                                                                                   training_data_used = training_data_utilized, 
                                                                                   val_data_used = val_data_utilized)

print('The metrics of best validation accuracy model is as follows.')
print(saved_models_metrics_obtained['best_val_acc_model'])

print('The metrics of best validation loss model is as follows.')
print(saved_models_metrics_obtained['best_val_loss_model'])

print('The metrics of the model from last epoch is as follows.')
print(saved_models_metrics_obtained['last_epoch_model'])


# Save the results from training into csv files.

train_val_record_file_name = 'NUMOResNet_train_val_record_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'RNAFold_data.csv'

df_train_val_record = pd.DataFrame.from_dict(train_val_record_obtained)

df_train_val_record.to_csv(train_val_record_file_name, index=False)

saved_models_metrics_file_name = 'NUMOResNet_saved_models_metrics_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'RNAFold_data.csv'

df_saved_models_metrics = pd.DataFrame.from_dict(saved_models_metrics_obtained)

df_saved_models_metrics.to_csv(saved_models_metrics_file_name, index=False)


#%%
# Load the trained models
# Best val accuracy model

best_val_acc_model_name_utilized = 'Best_validation_accuracy_model_'+model_name_utilized

NN_archit_test_best_acc = ResNet_18_pair_grayscale_mat_nt_localized_info_mat(nt_info_mat_input_num_channels=1, grayscale_mat_input_num_channels=4)

NN_test_model_best_acc = NN_archit_test_best_acc.to(device_utilized)

NN_test_model_best_acc.load_state_dict(torch.load(best_val_acc_model_name_utilized)['saved_model'])
print('The best validation accuracy model is saved from epoch ', torch.load(best_val_acc_model_name_utilized)['epoch_num'])

# Best val loss model

best_val_loss_model_name_utilized = 'Best_validation_loss_model_'+model_name_utilized

NN_archit_test_best_loss = ResNet_18_pair_grayscale_mat_nt_localized_info_mat(nt_info_mat_input_num_channels=1, grayscale_mat_input_num_channels=4)

NN_test_model_best_loss = NN_archit_test_best_loss.to(device_utilized)

NN_test_model_best_loss.load_state_dict(torch.load(best_val_loss_model_name_utilized)['saved_model'])
print('The best validation loss model is saved from epoch ', torch.load(best_val_loss_model_name_utilized)['epoch_num'])

# Model from last epoch

model_from_last_epoch_name_utilized = 'Model_from_last_epoch_'+model_name_utilized

NN_archit_test_model_last_epoch = ResNet_18_pair_grayscale_mat_nt_localized_info_mat(nt_info_mat_input_num_channels=1, grayscale_mat_input_num_channels=4)

NN_test_model_last_epoch = NN_archit_test_model_last_epoch.to(device_utilized)

NN_test_model_last_epoch.load_state_dict(torch.load(model_from_last_epoch_name_utilized)['saved_model'])
print('The model from last epoch is saved from epoch ', torch.load(model_from_last_epoch_name_utilized)['epoch_num'])


#%%
# Test trained models on testing data


loss_mathews_data_best_val_acc, metric_mathews_data_best_val_acc = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_acc, 
                                                                                           loss_function = loss_function_utilized, model_name = best_val_acc_model_name_utilized, 
                                                                                           testinging_data_used = testing_data_utilized, testing_data_name = 'Mathews_testing_data')

print('The metrics of best validation accuracy model on Mathews testing data is as follows.')
print(metric_mathews_data_best_val_acc)


loss_rfam_data_best_val_acc, metric_rfam_data_best_val_acc = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_acc, 
                                                                                           loss_function = loss_function_utilized, model_name = best_val_acc_model_name_utilized, 
                                                                                           testinging_data_used = rfam_testing_data_utilized, testing_data_name = 'Rfam_testing_data')

print('The metrics of best validation accuracy model on Rfam testing data is as follows.')
print(metric_rfam_data_best_val_acc)

loss_mathews_data_best_val_loss, metric_mathews_data_best_val_loss = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_loss, 
                                                                                           loss_function = loss_function_utilized, model_name = best_val_loss_model_name_utilized, 
                                                                                           testinging_data_used = testing_data_utilized, testing_data_name = 'Mathews_testing_data')

print('The metrics of best validation loss model on Mathews testing data is as follows.')
print(metric_mathews_data_best_val_loss)

loss_rfam_data_best_val_loss, metric_rfam_data_best_val_loss = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_loss, 
                                                                                           loss_function = loss_function_utilized, model_name = best_val_loss_model_name_utilized, 
                                                                                           testinging_data_used = rfam_testing_data_utilized, testing_data_name = 'Rfam_testing_data')

print('The metrics of best validation loss model on Rfam testing data is as follows.')
print(metric_rfam_data_best_val_loss)

loss_mathews_data_model_last_epoch, metric_mathews_data_model_last_epoch = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_last_epoch, 
                                                                                           loss_function = loss_function_utilized, model_name = model_from_last_epoch_name_utilized, 
                                                                                           testinging_data_used = testing_data_utilized, testing_data_name = 'Mathews_testing_data')

print('The metrics of the model from last epoch on Mathews testing data is as follows.')
print(metric_mathews_data_model_last_epoch)

loss_rfam_data_model_last_epoch, metric_rfam_data_model_last_epoch = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_last_epoch, 
                                                                                           loss_function = loss_function_utilized, model_name = model_from_last_epoch_name_utilized, 
                                                                                           testinging_data_used = rfam_testing_data_utilized, testing_data_name = 'Rfam_testing_data')

print('The metrics of the model from last epoch on Rfam testing data is as follows.')
print(metric_rfam_data_model_last_epoch)



#%% 

figure_name_utilized = 'NUMO_ResNet_18_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'_criteria_1_RNAFold_data'


train_val_figure_plot_function(figure_name = figure_name_utilized, 
                               loss_value_train = train_val_record_obtained['loss_train'], 
                               loss_value_val = train_val_record_obtained['loss_val'], 
                               accuracy_train = train_val_record_obtained['acc_train'], 
                               accuracy_val = train_val_record_obtained['acc_val'], 
                               auc_roc_train = train_val_record_obtained['aucroc_train'], 
                               auc_roc_val = train_val_record_obtained['aucroc_val'], 
                               auc_roc_train_check = train_val_record_obtained['aucroc_train_check'], 
                               auc_roc_val_check = train_val_record_obtained['aucroc_val_check'])
























