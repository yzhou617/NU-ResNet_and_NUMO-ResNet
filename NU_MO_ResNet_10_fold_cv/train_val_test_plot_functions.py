
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, roc_auc_score, matthews_corrcoef, precision_score, recall_score
import matplotlib.pyplot as plt
import copy


def create_weighted_sampler(y_numpy):
    
    print("The weighted sampler is being created.")
        
    train_y_for_weight_deploy = copy.deepcopy(y_numpy)
    num_label_1_train_data_deploy = len(train_y_for_weight_deploy[np.where(train_y_for_weight_deploy==1)])
    num_label_0_train_data_deploy = len(train_y_for_weight_deploy[np.where(train_y_for_weight_deploy==0)])
    weight_label_0_1_deploy = np.array([1/num_label_0_train_data_deploy, 1/num_label_1_train_data_deploy])
    weight_whole_train_data_deploy = np.where(train_y_for_weight_deploy == 0, weight_label_0_1_deploy[0], weight_label_0_1_deploy[1])
    sampler_weight_data_loader_deploy = torch.utils.data.sampler.WeightedRandomSampler(weight_whole_train_data_deploy, len(weight_whole_train_data_deploy))
    print('The shape of the training label data is ', y_numpy.shape)
    print('The weight for the sampler is ', weight_whole_train_data_deploy, ' and the corresponding size is ', weight_whole_train_data_deploy.shape)

    return sampler_weight_data_loader_deploy
    




def dataloader_prep_with_sampler(x_color_mat_stack_4D_tensor, x_nt_localized_info_mat_stack_4D_tensor, y_tensor, batch_size_value, sampler_applied):
    
    data_applied = torch.utils.data.TensorDataset(x_color_mat_stack_4D_tensor, x_nt_localized_info_mat_stack_4D_tensor, y_tensor)
    
    data_in_data_loader = torch.utils.data.DataLoader(dataset=data_applied, batch_size=batch_size_value, shuffle=False, sampler=sampler_applied)
    
    return data_in_data_loader





def dataloader_prep(x_color_mat_stack_4D_tensor, x_nt_localized_info_mat_stack_4D_tensor, y_tensor, batch_size_value):
    
    data_applied = torch.utils.data.TensorDataset(x_color_mat_stack_4D_tensor, x_nt_localized_info_mat_stack_4D_tensor, y_tensor)
    
    data_in_data_loader = torch.utils.data.DataLoader(dataset=data_applied, batch_size=batch_size_value, shuffle=False, sampler=None)
    
    return data_in_data_loader





def classification_model_performance_metrics(all_preds_score_positive_sample, all_preds_label, all_true_label):
    
    classification_metric = {'model_accuracy': float("-inf"), 
                             'model_auc_roc': float("-inf"),
                             'model_auc_roc_check': float("-inf"),
                             'model_mcc': float("-inf"),
                             'model_precision': float("-inf"),
                             'model_recall': float("-inf"),
                             'model_specificity': float("-inf")}
    
    print(classification_report(y_true=all_true_label, y_pred=all_preds_label))
    
    print('The confusion matrix is as follows.')
    confusion_mat_obtained = confusion_matrix(y_true=all_true_label, y_pred=all_preds_label)
    print(confusion_mat_obtained)
    
    accuracy_obtained = accuracy_score(y_true=all_true_label, y_pred=all_preds_label, normalize=True)
    print("The classification accuracy is {}.".format(accuracy_obtained))
    classification_metric['model_accuracy'] = accuracy_obtained
    
    false_positive_r, true_positive_r, threshold_obtained = roc_curve(y_true=all_true_label, y_score=all_preds_score_positive_sample, pos_label=1)
    auc_roc_obtained_check = auc(false_positive_r, true_positive_r)
    auc_roc_obtained = roc_auc_score(y_true=all_true_label, y_score=all_preds_score_positive_sample, sample_weight=None)
    print("The auc roc is {}.".format(auc_roc_obtained))
    print("The auc roc check is {}.".format(auc_roc_obtained_check))
    classification_metric['model_auc_roc'] = auc_roc_obtained
    classification_metric['model_auc_roc_check'] = auc_roc_obtained_check
    
    mcc_obtained = matthews_corrcoef(y_true=all_true_label, y_pred=all_preds_label, sample_weight=None)
    print("The mcc is {}.".format(mcc_obtained))
    classification_metric['model_mcc'] = mcc_obtained
    
    precision_obtained = precision_score(y_true=all_true_label, y_pred=all_preds_label, pos_label=1, average='binary', sample_weight=None)
    print("The precision is {}.".format(precision_obtained))
    classification_metric['model_precision'] = precision_obtained
    
    recall_obtained = recall_score(y_true=all_true_label, y_pred=all_preds_label, pos_label=1, average='binary', sample_weight=None)
    print("The recall is {}.".format(recall_obtained))
    classification_metric['model_recall'] = recall_obtained
    
    specificity_obtained = confusion_mat_obtained[0, 0]/(confusion_mat_obtained[0, 1]+confusion_mat_obtained[0, 0])
    print("The specificity is {}.".format(specificity_obtained))
    classification_metric['model_specificity'] = specificity_obtained
    
    return classification_metric





def train_val_ResNet_expert(device, NN_model, num_epochs, optimizer, scheduler_for_optimizer, loss_function, model_name, training_data_used, val_data_used):

    print("The training process of the model is starting.")
    loss_value_train = []
    loss_value_val = []
    accuracy_train = []
    accuracy_val = []
    auc_roc_train = []
    auc_roc_val = []
    auc_roc_check_train = []
    auc_roc_check_val = []
    mcc_train = []
    mcc_val = []
    precision_train = []
    precision_val = []
    recall_train = []
    recall_val = []
    specificity_train = []
    specificity_val = []
    
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_val_acc_model_metric = {'best_val_acc_model_accuracy': float("-inf"), 'best_val_acc_model_auc_roc': float("-inf")}
    best_val_loss_model_metric = {'best_val_loss_model_accuracy': float("-inf"), 'best_val_loss_model_auc_roc': float("-inf")}
    last_epoch_model_metric = {'last_epoch_model_accuracy': float("-inf"), 'last_epoch_model_auc_roc': float("-inf")}
    
    for epoch in range(num_epochs):
        print('Epoch {}:'.format(epoch+1))
        
        all_preds_train = []
        all_preds_score_train = []
        all_labels_train = []
        NN_model.train(mode=True)
        training_loss = 0
        
        for i, mat_label in enumerate(training_data_used):
            
            color_mat_train, nt_localized_info_mat_train, label_train = mat_label[0].to(device), mat_label[1].to(device), mat_label[2].to(device)
            optimizer.zero_grad() 
            pred_train = NN_model(color_mat_train, nt_localized_info_mat_train)
            loss = loss_function(pred_train, label_train)
            loss.backward()  
            optimizer.step()  
            
            pred_train_score = torch.nn.functional.softmax(pred_train.cpu().detach(), dim=1)
    
            all_preds_score_train.append(pred_train_score.numpy()[:, 1])
            
            all_preds_train.append(np.argmax(pred_train.cpu().detach().numpy(), axis=1))
            
            all_labels_train.append(label_train.cpu().detach().numpy())
            training_loss += loss.item()
        
        all_preds_score_train = np.concatenate(all_preds_score_train).ravel()
        all_preds_train = np.concatenate(all_preds_train).ravel()
        all_labels_train = np.concatenate(all_labels_train).ravel()
    
        print("The results from training are as follows.")
        
        classification_metric_train = classification_model_performance_metrics(all_preds_score_positive_sample = all_preds_score_train, 
                                                                               all_preds_label = all_preds_train, 
                                                                               all_true_label = all_labels_train)
        
        accuracy_train.append(classification_metric_train['model_accuracy'])
        auc_roc_train.append(classification_metric_train['model_auc_roc'])
        auc_roc_check_train.append(classification_metric_train['model_auc_roc_check'])
        mcc_train.append(classification_metric_train['model_mcc'])
        precision_train.append(classification_metric_train['model_precision'])
        recall_train.append(classification_metric_train['model_recall'])
        specificity_train.append(classification_metric_train['model_specificity'])
        
        training_loss_this_epoch = training_loss/len(training_data_used)
        print('The training loss value is {}.'.format(training_loss_this_epoch))
        loss_value_train.append(training_loss_this_epoch)
        
        
        all_preds_val = []
        all_preds_score_val = []
        all_labels_val = []
        NN_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, mat_label in enumerate(val_data_used):
                
                color_mat_val, nt_localized_info_mat_val, label_val = mat_label[0].to(device), mat_label[1].to(device), mat_label[2].to(device)
                pred_val = NN_model(color_mat_val, nt_localized_info_mat_val)
                loss = loss_function(pred_val, label_val)
                pred_val_score = torch.nn.functional.softmax(pred_val.cpu().detach(), dim=1)
                
                all_preds_score_val.append(pred_val_score.numpy()[:, 1])
                all_preds_val.append(np.argmax(pred_val.cpu().detach().numpy(), axis=1))
                all_labels_val.append(label_val.cpu().detach().numpy())
                val_loss += loss.item()
        
        print("The performance of the current model based on the validation data is as follows.")
        all_preds_score_val = np.concatenate(all_preds_score_val).ravel()
        all_preds_val = np.concatenate(all_preds_val).ravel()
        all_labels_val = np.concatenate(all_labels_val).ravel()
        
        classification_metric_val = classification_model_performance_metrics(all_preds_score_positive_sample = all_preds_score_val, 
                                                                             all_preds_label = all_preds_val, 
                                                                             all_true_label = all_labels_val)
        
        accuracy_val.append(classification_metric_val['model_accuracy'])
        auc_roc_val.append(classification_metric_val['model_auc_roc'])
        auc_roc_check_val.append(classification_metric_val['model_auc_roc_check'])
        mcc_val.append(classification_metric_val['model_mcc'])
        precision_val.append(classification_metric_val['model_precision'])
        recall_val.append(classification_metric_val['model_recall'])
        specificity_val.append(classification_metric_val['model_specificity'])
        
        
        val_loss_this_epoch = val_loss/len(val_data_used)
        print('The validation loss value is {}.'.format(val_loss_this_epoch))
        loss_value_val.append(val_loss_this_epoch)
        
        scheduler_for_optimizer.step()
        
        best_val_acc_model_name = 'Best_validation_accuracy_model_'+model_name
        
        if classification_metric_val['model_accuracy'] > best_val_acc:
            best_val_acc = classification_metric_val['model_accuracy']
            best_val_acc_model_metric['best_val_acc_model_accuracy'] = classification_metric_val['model_accuracy']
            best_val_acc_model_metric['best_val_acc_model_auc_roc'] = classification_metric_val['model_auc_roc']
            best_val_acc_model_metric['best_val_acc_model_auc_roc_check'] = classification_metric_val['model_auc_roc_check']
            best_val_acc_model_metric['best_val_acc_model_mcc'] = classification_metric_val['model_mcc']
            best_val_acc_model_metric['best_val_acc_model_precision'] = classification_metric_val['model_precision']
            best_val_acc_model_metric['best_val_acc_model_recall'] = classification_metric_val['model_recall']
            best_val_acc_model_metric['best_val_acc_model_specificity'] = classification_metric_val['model_specificity']
            
            print('The current best validation accuracy becomes ', best_val_acc)
            torch.save({'epoch_num': epoch+1,
                        'saved_loss_function': loss_function,
                        'saved_optimizer': optimizer.state_dict(),
                        'saved_model': NN_model.state_dict()},
                       best_val_acc_model_name)
        else:
            print('The current best validation accuracy does not change.')
        
        best_val_loss_model_name = 'Best_validation_loss_model_' + model_name
        
        if val_loss_this_epoch < best_val_loss:
            best_val_loss = val_loss_this_epoch
            best_val_loss_model_metric['best_val_loss_model_accuracy'] = classification_metric_val['model_accuracy']
            best_val_loss_model_metric['best_val_loss_model_auc_roc'] = classification_metric_val['model_auc_roc']
            best_val_loss_model_metric['best_val_loss_model_auc_roc_check'] = classification_metric_val['model_auc_roc_check']
            best_val_loss_model_metric['best_val_loss_model_mcc'] = classification_metric_val['model_mcc']
            best_val_loss_model_metric['best_val_loss_model_precision'] = classification_metric_val['model_precision']
            best_val_loss_model_metric['best_val_loss_model_recall'] = classification_metric_val['model_recall']
            best_val_loss_model_metric['best_val_loss_model_specificity'] = classification_metric_val['model_specificity']
            
            print('The current best validation loss becomes ', best_val_loss)
            torch.save({'epoch_num': epoch+1,
                        'saved_loss_function': loss_function,
                        'saved_optimizer': optimizer.state_dict(),
                        'saved_model': NN_model.state_dict()},
                       best_val_loss_model_name)
        else:
            print('The current best validation loss does not change.')
    
    model_name_from_last_epoch = 'Model_from_last_epoch_' + model_name
    
    last_epoch_model_metric['last_epoch_model_accuracy'] = classification_metric_val['model_accuracy']
    last_epoch_model_metric['last_epoch_model_auc_roc'] = classification_metric_val['model_auc_roc']
    last_epoch_model_metric['last_epoch_model_auc_roc_check'] = classification_metric_val['model_auc_roc_check']
    last_epoch_model_metric['last_epoch_model_mcc'] = classification_metric_val['model_mcc']
    last_epoch_model_metric['last_epoch_model_precision'] = classification_metric_val['model_precision']
    last_epoch_model_metric['last_epoch_model_recall'] = classification_metric_val['model_recall']
    last_epoch_model_metric['last_epoch_model_specificity'] = classification_metric_val['model_specificity']
    
    torch.save({'epoch_num': num_epochs,
                'saved_loss_function': loss_function,
                'saved_optimizer': optimizer.state_dict(),
                'saved_model': NN_model.state_dict()},
               model_name_from_last_epoch)
    
    train_val_record = dict({'loss_train': loss_value_train, 
                             'loss_val': loss_value_val, 
                             'acc_train': accuracy_train, 
                             'acc_val': accuracy_val, 
                             'aucroc_train': auc_roc_train, 
                             'aucroc_val': auc_roc_val, 
                             'aucroc_train_check': auc_roc_check_train, 
                             'aucroc_val_check': auc_roc_check_val})
    
    saved_models_metrics = {'best_val_acc_model': best_val_acc_model_metric, 
                            'best_val_loss_model': best_val_loss_model_metric, 
                            'last_epoch_model': last_epoch_model_metric}
    
    return train_val_record, saved_models_metrics





def test_ResNet_expert(device, NN_model, loss_function, model_name, testinging_data_used, testing_data_name):
    
    all_preds_test = []
    all_preds_score_test = []
    all_labels_test = []
    NN_model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for i, mat_label in enumerate(testinging_data_used):
            color_mat_test, nt_localized_info_mat_test, label_test = mat_label[0].to(device), mat_label[1].to(device), mat_label[2].to(device)
            pred_test = NN_model(color_mat_test, nt_localized_info_mat_test)
            loss = loss_function(pred_test, label_test)
            pred_test_score = torch.nn.functional.softmax(pred_test.cpu().detach(), dim=1)
            
            all_preds_score_test.append(pred_test_score.numpy()[:, 1])
            
            all_preds_test.append(np.argmax(pred_test.cpu().detach().numpy(), axis=1))
            all_labels_test.append(label_test.cpu().detach().numpy())
            test_loss += loss.item()
        
    print("The performance of the trained model ",model_name," based on the ",testing_data_name," is as follows.")
    
    all_preds_score_test = np.concatenate(all_preds_score_test).ravel()
    all_preds_test = np.concatenate(all_preds_test).ravel()
    all_labels_test = np.concatenate(all_labels_test).ravel()
    
    classification_metric_test = classification_model_performance_metrics(all_preds_score_positive_sample = all_preds_score_test, 
                                                                          all_preds_label = all_preds_test, 
                                                                          all_true_label = all_labels_test)

    test_loss_this_data_set = test_loss/len(testinging_data_used)
    print('The testing loss value is {}.'.format(test_loss_this_data_set))
    
    return test_loss_this_data_set, classification_metric_test





def train_val_figure_plot_function(figure_name, loss_value_train, loss_value_val, accuracy_train, accuracy_val, auc_roc_train, auc_roc_val, auc_roc_train_check, auc_roc_val_check):
    
    x_index = list(range(len(loss_value_train)))
    x_index_utilized = list(range(1, len(loss_value_train)+1))

    figure_name_loss_value_train_val = figure_name + '_loss_value_train_validation_data.jpg'
    figure_name_accuracy_train_val = figure_name + '_accuracy_train_validation_data.jpg'
    figure_name_accuracy_val = figure_name + '_accuracy_validation_data.jpg'
    figure_name_auc_roc_train_val = figure_name + '_auc_roc_train_validation_data.jpg'
    figure_name_auc_roc_train_val_check = figure_name + '_auc_roc_train_validation_data_check.jpg'
    
    plt.figure(num=1, figsize=(18, 16))
    plt.plot(x_index, loss_value_train, color='blue', label='Training loss')
    plt.plot(x_index, loss_value_val, color='darkorange', label='Validation loss')
    plt.xticks(x_index, x_index_utilized)
    plt.title('The training loss and validation loss', fontsize=22)
    plt.legend()
    plt.xlabel('number of epochs', fontsize=20)
    plt.ylabel('loss value', fontsize=20)
    plt.ylim(0, 3.5)
    plt.savefig(figure_name_loss_value_train_val)
    
    plt.figure(num=2, figsize=(18, 16))
    plt.plot(x_index, accuracy_train, color='blue', label='Accuracy on training data')
    plt.plot(x_index, accuracy_val, color='darkorange', label='Accuracy on validation data')
    plt.xticks(x_index, x_index_utilized)
    plt.title('The accuracy based on training data and validation data', fontsize=22)
    plt.legend()
    plt.xlabel('number of epochs', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.ylim(0, 1)
    plt.savefig(figure_name_accuracy_train_val)
    
    plt.figure(num=3, figsize=(18, 16))
    plt.plot(x_index, accuracy_val, color='darkorange', label='Accuracy on validation data')
    plt.xticks(x_index, x_index_utilized)
    plt.title('The accuracy based on validation data', fontsize=22)
    plt.legend()
    plt.xlabel('number of epochs', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.ylim(0, 1)
    plt.savefig(figure_name_accuracy_val)    
    
    plt.figure(num=4, figsize=(18, 16))
    plt.plot(x_index, auc_roc_train, color='blue', label='AUC ROC on training data')
    plt.plot(x_index, auc_roc_val, color='darkorange', label='AUC ROC on validation data')
    plt.xticks(x_index, x_index_utilized)
    plt.title('The AUC ROC based on training and validation data', fontsize=22)
    plt.legend()
    plt.xlabel('number of epochs', fontsize=20)
    plt.ylabel('AUC ROC', fontsize=20)
    plt.ylim(0, 1)
    plt.savefig(figure_name_auc_roc_train_val)
    
    plt.figure(num=5, figsize=(18, 16))
    plt.plot(x_index, auc_roc_train_check, color='blue', label='AUC ROC on training data check')
    plt.plot(x_index, auc_roc_val_check, color='darkorange', label='AUC ROC on validation data check')
    plt.xticks(x_index, x_index_utilized)
    plt.title('The AUC ROC based on training and validation data check', fontsize=22)
    plt.legend()
    plt.xlabel('number of epochs', fontsize=20)
    plt.ylabel('AUC ROC', fontsize=20)
    plt.ylim(0, 1)
    plt.savefig(figure_name_auc_roc_train_val_check)

















