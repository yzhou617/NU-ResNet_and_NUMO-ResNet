
import numpy as np
import pandas as pd
from RNA_motif_info_extraction_function_v10 import RNA_motif_info_extraction
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import norm



def nt_localized_info_mat_generator(RNA_seq, RNA_ss):
    
    # 1. Obtain the motif information matrix by utilizing the function RNA_motif_info_extraction
    motif_info_mat = RNA_motif_info_extraction(RNA_seq, RNA_ss)
    
    motif_info_mat = motif_info_mat[['nt','motif_1','motif_1_FE','motif_2','motif_2_FE']]
    
    # 2. Transform motif_1 and motif_2 to one hot encoding
    
    # 2.1 construct the transformer
    motif_type_list = ['stack', 'hairpin_loop', 'interior_loop', 'bulge_loop', 'bifurcation_loop', 'NONE']
    
    #motif_type_list = list(np.array(['stack', 'hairpin_loop', 'interior_loop', 'bulge_loop', 'bifurcation_loop', 'missing_value']).reshape(1,6))
    
    
    motif_transform = Pipeline(steps=[('tackle_nan', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NONE')),
                                      ('motif_ohe', OneHotEncoder(categories=[motif_type_list], sparse=False, handle_unknown='error'))])
    
    nt_ohe = OneHotEncoder(categories=[['A','U','C','G']], sparse=False, handle_unknown='error')
    
    data_processor = ColumnTransformer(transformers = [('nt_tf', nt_ohe, ['nt']),
                                                       ('motif_1_tf', motif_transform, ['motif_1']),
                                                       ('motif_2_tf', motif_transform, ['motif_2'])])
    
    nt_motif_info_mat_np = data_processor.fit_transform(motif_info_mat)
    
    nt_motif_ohe = pd.DataFrame(nt_motif_info_mat_np)
    
    nt_motif_ohe.columns = ['A','U','C','G',
                            'stack_motif_1', 'hairpin_loop_motif_1', 'interior_loop_motif_1', 'bulge_loop_motif_1', 'bifurcation_loop_motif_1', 'NONE_motif_1',
                            'stack_motif_2', 'hairpin_loop_motif_2', 'interior_loop_motif_2', 'bulge_loop_motif_2', 'bifurcation_loop_motif_2', 'NONE_motif_2']
    
    motif_info_mat = motif_info_mat.join(nt_motif_ohe)
    #motif_info_mat.to_csv("motif_info_mat_test.csv",index=False)
    
    del motif_info_mat["nt"]
    del motif_info_mat["motif_1"]
    del motif_info_mat["motif_2"]
    
    motif_info_mat = motif_info_mat.reindex(columns=['A', 'U', 'C', 'G',
                                                     'stack_motif_1', 'hairpin_loop_motif_1', 'interior_loop_motif_1',
                                                     'bulge_loop_motif_1', 'bifurcation_loop_motif_1', 'NONE_motif_1', 'motif_1_FE',
                                                     'stack_motif_2', 'hairpin_loop_motif_2', 'interior_loop_motif_2',
                                                     'bulge_loop_motif_2', 'bifurcation_loop_motif_2', 'NONE_motif_2', 'motif_2_FE'])
    
    
    motif_info_mat['motif_1_FE'] = motif_info_mat['motif_1_FE']/100
    motif_info_mat['motif_2_FE'] = motif_info_mat['motif_2_FE']/100
    
    motif_info_mat['motif_1_FE'] = norm.cdf(motif_info_mat['motif_1_FE'], loc=0, scale=5)
    motif_info_mat['motif_2_FE'] = norm.cdf(motif_info_mat['motif_2_FE'], loc=0, scale=5)
    
    
    motif_info_mat_np_arr = motif_info_mat.to_numpy().astype('float64')
        
    return motif_info_mat_np_arr
        

# The following example is for testing the function. 
'''
RNA_seq = "AACGGUCCAACGGAGUGUAACCGCCCGU"
RNA_ss = ".(((((...))((.((...))..)))))"
nt_localized_info_mat_test = nt_localized_info_mat_generator(RNA_seq, RNA_ss)
print('The testing nt localized info mat is', nt_localized_info_mat_test)
print('The shape of the testing nt localized info mat is ', nt_localized_info_mat_test.shape)
'''













