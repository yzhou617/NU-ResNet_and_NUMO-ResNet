
import numpy as np
import pandas as pd
from rna_tools.SecondaryStructure import parse_vienna_to_pairs


########## Define the funtion to generate the color matrix for one pair of RNA Sequence and RNA Secondary Structure

def grayscale_matrix_generator(RNA_seq, RNA_ss):
    #  2.1 obtain the index for the paired bases 
    parse_vienna_to_pairs(RNA_ss)
    pairs_index, pairs_pk_index = parse_vienna_to_pairs(RNA_ss)
    
    # 2.2 construct the colormap matrix 
    RNA_seq_split = list(RNA_seq)
    grayscale_mat_c_1 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_2 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_3 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_4 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    
    # 2.3 assign value to each element of the colormap matrix
    row_name_colorm = list(grayscale_mat_c_1.index)
    col_name_colorm = list(grayscale_mat_c_1.columns)
    
    # 3.1 In the diagonal, assign vector to each nucleotide
    for i, row_base in enumerate(row_name_colorm, start=1):
        if row_base == 'A':
            grayscale_mat_c_1.iloc[i-1, i-1] = 1
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'U':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 1
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'C':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 1
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'G':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 1
        
    
    # 3.2 For the paired nucleotides, assign different vectors to represent differnt types of base pairing
    for k, paired_base_index in enumerate(pairs_index, start=1):
        paired_base_index_x = paired_base_index[0] - 1
        paired_base_index_y = paired_base_index[1] - 1
    
        if row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
    
    
    for l, paired_base_index_2 in enumerate(pairs_index, start=1):
        paired_base_index_x = paired_base_index_2[1] - 1
        paired_base_index_y = paired_base_index_2[0] - 1
    
        if row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
    
    # 3.3 For the unpaired nucleotides, fill in NAN with 0
    grayscale_mat_c_1 = grayscale_mat_c_1.fillna(0)
    grayscale_mat_c_2 = grayscale_mat_c_2.fillna(0)
    grayscale_mat_c_3 = grayscale_mat_c_3.fillna(0)
    grayscale_mat_c_4 = grayscale_mat_c_4.fillna(0)
    
    grayscale_mat_c_1_np_arr = grayscale_mat_c_1.to_numpy().astype('int64')
    grayscale_mat_c_2_np_arr = grayscale_mat_c_2.to_numpy().astype('int64')
    grayscale_mat_c_3_np_arr = grayscale_mat_c_3.to_numpy().astype('int64')
    grayscale_mat_c_4_np_arr = grayscale_mat_c_4.to_numpy().astype('int64')
    
    grayscale_mat = np.array([grayscale_mat_c_1_np_arr, grayscale_mat_c_2_np_arr, 
                              grayscale_mat_c_3_np_arr, grayscale_mat_c_4_np_arr])
    
    return grayscale_mat
    

##########

