
import numpy as np
import pandas as pd
from grayscale_matrix_generation_function import grayscale_matrix_generator
from nt_localized_info_matrix_generation_function import nt_localized_info_mat_generator



def data_prep_grayscale_mat_nt_localized_info_mat(Original_RNA_Data):
    
    # 1.1.1 Shuffle the data
    Original_RNA_Data = Original_RNA_Data.sample(frac=1, replace=False, random_state=7).reset_index(drop=True)
    
    # 1. Create columns and obtain the shape of the Original_RNA_Data
    Original_RNA_Data['Normalized_MCC']=np.nan
    Original_RNA_Data['color_matrix_seq_correctss']=np.nan
    Original_RNA_Data['color_matrix_seq_exprtss']=np.nan
    Original_RNA_Data['nt_localized_info_matrix_seq_correctss']=np.nan
    Original_RNA_Data['nt_localized_info_matrix_seq_exprtss']=np.nan
    
    Original_RNA_Data['color_matrix_seq_correctss'] = Original_RNA_Data['color_matrix_seq_correctss'].astype(object)
    Original_RNA_Data['color_matrix_seq_exprtss'] = Original_RNA_Data['color_matrix_seq_exprtss'].astype(object)
    
    Original_RNA_Data['nt_localized_info_matrix_seq_correctss'] = Original_RNA_Data['nt_localized_info_matrix_seq_correctss'].astype(object)
    Original_RNA_Data['nt_localized_info_matrix_seq_exprtss'] = Original_RNA_Data['nt_localized_info_matrix_seq_exprtss'].astype(object)
    
    # 2.1 Obtain the number of rows and columns of the data frame
    num_row_Original_RNA_Data = Original_RNA_Data.shape[0]
    num_col_Original_RNA_Data = Original_RNA_Data.shape[1]
    
    # 2.2 Genarate a grayscale color matrix and nt localized info matrix for each pair of RNA Sequence and RNA Secondary Structure
    for i in range(num_row_Original_RNA_Data):
        Original_RNA_Data.at[i, 'color_matrix_seq_correctss']=grayscale_matrix_generator(Original_RNA_Data.iloc[i,1], Original_RNA_Data.iloc[i,2])
        Original_RNA_Data.at[i, 'color_matrix_seq_exprtss']=grayscale_matrix_generator(Original_RNA_Data.iloc[i,1], Original_RNA_Data.iloc[i,3])
        Original_RNA_Data.at[i, 'nt_localized_info_matrix_seq_correctss']=nt_localized_info_mat_generator(Original_RNA_Data.iloc[i,1], Original_RNA_Data.iloc[i,2])
        Original_RNA_Data.at[i, 'nt_localized_info_matrix_seq_exprtss']=nt_localized_info_mat_generator(Original_RNA_Data.iloc[i,1], Original_RNA_Data.iloc[i,3])
    
    # 2.3 Normalize the MCC
    Original_RNA_Data['Normalized_MCC'] = (Original_RNA_Data['MCC']+1)/2
    
    # For each RNA sample, the code generates two grayscale color matrix and two nt localized info matrix. One grayscale color matrix and one nt localized info matrix are from actual RNA secondary structure. Another
    # one grayscale color matrix and another one nt localized info matrix are from predicted RNA secondary structure.  
    
    # 2.3.1.1 Obtain the grayscale color matrix and nt localized info matrix from the actual RNA secondary structure
    color_mat_3d_actual_RNA_structure = Original_RNA_Data[['color_matrix_seq_correctss', 'nt_localized_info_matrix_seq_correctss']].copy(deep=True)
    
    # 2.3.1.2 Add one column normalized MCC score for the corresponding grayscale color matrix 
    color_mat_3d_actual_RNA_structure['Normalized_MCC']=np.nan
    color_mat_3d_actual_RNA_structure['Normalized_MCC']=1
    
    # 2.3.2.1 Obtain the grayscale color matrix and nt localized info matrix from the predicted RNA secondary structure
    color_mat_3d_predicted_RNA_structure = Original_RNA_Data[['color_matrix_seq_exprtss', 'nt_localized_info_matrix_seq_exprtss', 'Normalized_MCC']].copy(deep=True)
    
    # 2.3.2.2 Keep the grayscale color matrix whose corresponding normalized MCC is less than 1
    color_mat_3d_predicted_RNA_structure = color_mat_3d_predicted_RNA_structure[color_mat_3d_predicted_RNA_structure['Normalized_MCC'] < 1]
    
    # 2.3.3 Rename the corresponding column name
    color_mat_3d_actual_RNA_structure = color_mat_3d_actual_RNA_structure.rename(columns={"color_matrix_seq_correctss":"color_matrix_utilized", "nt_localized_info_matrix_seq_correctss":"nt_localized_info_matrix_utilized"}) 
    
    color_mat_3d_predicted_RNA_structure = color_mat_3d_predicted_RNA_structure.rename(columns={"color_matrix_seq_exprtss":"color_matrix_utilized", "nt_localized_info_matrix_seq_exprtss":"nt_localized_info_matrix_utilized"}) 
    
    # 2.3.4 Combine two data sets
    color_matrix_data_set_combined = pd.concat([color_mat_3d_actual_RNA_structure, color_mat_3d_predicted_RNA_structure], axis=0, ignore_index=True)
    
    # 2.5 Create the label column: 1 represents the positive sample and 0 represents the negative sample
    color_matrix_data_set_combined['RNA_Label'] = np.where(color_matrix_data_set_combined['Normalized_MCC'] < 1, 0, 1)
    
    # 2.6 Obtain the number of rows in the color_matrix_data_set_combined
    num_row_color_matrix_data_set_combined = color_matrix_data_set_combined.shape[0]
    
    # 3.1 Obatin the size of each grayscale color matrix
    color_matrix_data_set_combined['color_mat_size'] = np.nan
    
    for i in range(num_row_color_matrix_data_set_combined):
        color_matrix_data_set_combined.at[i, 'color_mat_size'] = color_matrix_data_set_combined.iloc[i,0].shape[2]
    
    # Obtain the maximum size of all grayscale color matrix and do the padding for all grayscale color matrix and all nt localized info matrix
    size_maximum = 410.0
    
    print("The maximum length of the RNA sequences is {}.".format(size_maximum))
    
    color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'] = np.nan
    color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'] = color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'].astype(object)
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'] = np.nan
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'] = color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'].astype(object)
    
    for i in range(num_row_color_matrix_data_set_combined):
        size_diff = size_maximum - color_matrix_data_set_combined.at[i, 'color_mat_size']
        size_diff_utilized = int(size_diff)
        color_matrix_data_set_combined.at[i, 'nt_localized_info_matrix_utilized_for_CNN_padding']=np.pad(color_matrix_data_set_combined.iloc[i,1], ((0, size_diff_utilized), (0, 0)), 'constant')
        
        if (size_diff % 2) == 0:
            num_padding_left_above = int(size_diff/2)
            num_padding_right_below = int(size_diff/2)
        else:
            num_padding_left_above = int(size_diff/2)
            num_padding_right_below = int(size_diff/2)+1
        
        color_matrix_data_set_combined.at[i, 'color_matrix_utilized_for_CNN_padding']=np.pad(color_matrix_data_set_combined.iloc[i,0], ((0, 0), (num_padding_left_above, num_padding_right_below), (num_padding_left_above, num_padding_right_below)), 'constant')
    
    # 3.2 Test the size of the grayscale color matrix after do the padding
    color_matrix_data_set_combined['color_mat_size_test_af_pad'] = np.nan
    color_matrix_data_set_combined['nt_localized_info_mat_size_test_af_pad'] = np.nan
    
    for i in range(num_row_color_matrix_data_set_combined):
        color_matrix_data_set_combined.at[i, 'color_mat_size_test_af_pad'] = color_matrix_data_set_combined.at[i, 'color_matrix_utilized_for_CNN_padding'].shape[2]
        color_matrix_data_set_combined.at[i, 'nt_localized_info_mat_size_test_af_pad'] = color_matrix_data_set_combined.at[i, 'nt_localized_info_matrix_utilized_for_CNN_padding'].shape[0]
    
    # 3.3 Shuffle the data
    color_matrix_data_set_combined = color_matrix_data_set_combined.sample(frac=1, random_state=59).reset_index(drop=True)
    
    # 4.1 Obtain the grayscale color matrix and label for each RNA samples. And transfer the df to np
    x = color_matrix_data_set_combined[['color_matrix_utilized_for_CNN_padding', 'nt_localized_info_matrix_utilized_for_CNN_padding']]
    y = color_matrix_data_set_combined[['RNA_Label']]
    
    train_x_color_mat = x.loc[:, 'color_matrix_utilized_for_CNN_padding'].to_numpy()
    
    train_x_nt_localized_info_mat = x.loc[:, 'nt_localized_info_matrix_utilized_for_CNN_padding'].to_numpy()
    
    train_y = y.iloc[:,0].to_numpy()
    
    # 4.2 Transfer the 1-D np array to 3-D np array
    train_x_color_mat_stack = train_x_color_mat[0][np.newaxis,:,:,:]
    
    for j, mat in enumerate(train_x_color_mat[1:]):
        color_mat_augmented = mat[np.newaxis,:,:,:]
        train_x_color_mat_stack = np.vstack((train_x_color_mat_stack, color_mat_augmented))
    
    train_x_nt_localized_info_mat_stack = train_x_nt_localized_info_mat[0][np.newaxis,np.newaxis,:,:]
        
    for j, mat in enumerate(train_x_nt_localized_info_mat[1:]):
        nt_localized_info_mat_augmented = mat[np.newaxis,np.newaxis,:,:]
        train_x_nt_localized_info_mat_stack = np.vstack((train_x_nt_localized_info_mat_stack, nt_localized_info_mat_augmented))
    
    
    return train_x_color_mat_stack, train_x_nt_localized_info_mat_stack, train_y























