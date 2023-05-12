
import pandas as pd
import numpy as np
from rna_tools.SecondaryStructure import parse_vienna_to_pairs
import RNA


def RNA_motif_info_extraction(RNA_seq, RNA_ss):
    # 1. Obtain the index of base pairs.
    pairs_index, pairs_pk_index = parse_vienna_to_pairs(RNA_ss)
    
    
    # 2. Construct the index column as well as split the RNA sequence and RNA secondary structure   
    
    RNA_len = len(RNA_seq)
    index_list = list(range(1,RNA_len+1))
    
    RNA_seq_split = list(RNA_seq)
    
    RNA_ss_split = list(RNA_ss)
    
    # 3. Construct the nt based data frame for one RNA
    
    motif_info_data_initial = {'index': index_list, 'nt': RNA_seq_split, 'structure': RNA_ss_split}
    
    motif_info_data_df = pd.DataFrame(data=motif_info_data_initial)
    
    
    # 4. Assign the base pair into the motif_info_data_df, determine the kinds of motifs of RNA, and keep the index of motifs
    
    motif_info_data_df['base_pair_index']=np.nan
    motif_info_data_df['base_pair_index'] = motif_info_data_df['base_pair_index'].astype(object)
    
    motif_info_data_df['motif_1']=np.nan
    motif_info_data_df['motif_1'] = motif_info_data_df['motif_1'].astype(object)
    
    motif_info_data_df['motif_1_seq']=np.nan
    motif_info_data_df['motif_1_seq'] = motif_info_data_df['motif_1_seq'].astype(object)
    
    motif_info_data_df['motif_1_index']=np.nan
    motif_info_data_df['motif_1_index'] = motif_info_data_df['motif_1_index'].astype(object)
    
    
    # 4.1 Determine the types of motifs
    for base_pair in pairs_index:
        
        motif_info_data_df.at[base_pair[0]-1, 'base_pair_index'] = base_pair
        motif_info_data_df.at[base_pair[1]-1, 'base_pair_index'] = base_pair
     
        # 4.1.1 determine the stack 
        if motif_info_data_df.at[base_pair[0], 'structure'] == "(" and\
            motif_info_data_df.at[base_pair[1]-2, 'structure'] == ")" and\
                [base_pair[0]+1, base_pair[1]-1] in pairs_index and\
                    not np.isnan(motif_info_data_df.at[base_pair[0]-1, 'base_pair_index']).any() and\
                        not np.isnan(motif_info_data_df.at[base_pair[1]-1, 'base_pair_index']).any():
                            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "stack"
                            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "stack"
                            
                            motif_1_sequence = motif_info_data_df.at[base_pair[0]-1, 'nt']+\
                                motif_info_data_df.at[base_pair[0], 'nt']+\
                                    motif_info_data_df.at[base_pair[1]-2, 'nt']+\
                                        motif_info_data_df.at[base_pair[1]-1, 'nt']
                            
                            motif_info_data_df.at[base_pair[0]-1, 'motif_1_seq'] = motif_1_sequence
                            motif_info_data_df.at[base_pair[1]-1, 'motif_1_seq'] = motif_1_sequence
        # 4.1.2 determine the bulge/interior/bifurcation loop and hairpin loop                    
        elif motif_info_data_df.at[base_pair[0], 'structure'] == "." or\
            motif_info_data_df.at[base_pair[1]-2, 'structure'] == ".":
                str_within_loop = []
                loop_length = base_pair[1]-base_pair[0]
                for i in range(loop_length-1):
                    str_within_loop.append(motif_info_data_df.at[base_pair[0]+i, 'structure'])
                  
                if "(" in str_within_loop or ")" in str_within_loop:
                    motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bul_int_bir_loop"
                    motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bul_int_bir_loop"
                else:
                    motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "hairpin_loop"
                    motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "hairpin_loop"
                    
                    motif_1_sequence = motif_info_data_df.at[base_pair[0]-1, 'nt']
                    for i in range(loop_length):
                        motif_1_sequence += motif_info_data_df.at[base_pair[0]+i, 'nt']
                        
                    motif_info_data_df.at[base_pair[0]-1, 'motif_1_seq'] = motif_1_sequence
                    motif_info_data_df.at[base_pair[1]-1, 'motif_1_seq'] = motif_1_sequence
        # 4.1.3 determine the bifurcation loop
        #else:
            #motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bifurcation_loop"
            #motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bifurcation_loop"
            
                          
    
    # 4.2 Determine the index of motifs
    for base_pair in pairs_index:
        
        # 4.2.1 determine the index of the stack
        if motif_info_data_df.at[base_pair[0]-1, 'motif_1'] == "stack" and motif_info_data_df.at[base_pair[1]-1, 'motif_1'] == "stack":
            motif_info_data_df.at[base_pair[0]-1, 'motif_1_index'] = [motif_info_data_df.at[base_pair[0]-1, 'base_pair_index'][0],
                                                                      motif_info_data_df.at[base_pair[0], 'base_pair_index'][0],
                                                                      motif_info_data_df.at[base_pair[1]-2, 'base_pair_index'][1],
                                                                      motif_info_data_df.at[base_pair[1]-1, 'base_pair_index'][1]]
            
            motif_info_data_df.at[base_pair[1]-1, 'motif_1_index'] = motif_info_data_df.at[base_pair[0]-1, 'motif_1_index']
        
        # 4.2.2 determine the index of hairpin loop                    
        elif motif_info_data_df.at[base_pair[0]-1, 'motif_1'] == "hairpin_loop" and motif_info_data_df.at[base_pair[1]-1, 'motif_1'] == "hairpin_loop":
            loop_index = []
            loop_length = base_pair[1]-base_pair[0]
            for i in range(loop_length + 1):
                loop_index.append(motif_info_data_df.at[base_pair[0]-1+i, 'index'])
                
            motif_info_data_df.at[base_pair[0]-1, 'motif_1_index'] = loop_index
            motif_info_data_df.at[base_pair[1]-1, 'motif_1_index'] = motif_info_data_df.at[base_pair[0]-1, 'motif_1_index']
                                
    
    # 5. Distinguish bifurcation loop from bulge/interior loop
    
    motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs']=np.nan
    motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs'] = motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs'].astype(object)
    
    # 5.1 obtain the index of base pairs whose motif_1 equals to bul_int_bir_loop 
    motif_info_data_df_temp = motif_info_data_df[motif_info_data_df["motif_1"]=="bul_int_bir_loop"].reset_index(drop=True)
    
    num_rows_motif_info_data_df_temp = motif_info_data_df_temp.shape[0]
    
    temp_base_pair_index = []
    
    for i in range(num_rows_motif_info_data_df_temp):
        if motif_info_data_df_temp.at[i, 'base_pair_index'] not in temp_base_pair_index:
            temp_base_pair_index.append(motif_info_data_df_temp.at[i, 'base_pair_index'])
    '''
    # 5.2 distinguish bifurcation loop from bulge/interior loop  
    for base_pair in temp_base_pair_index:
        loop_length = base_pair[1]-base_pair[0]
        bul_int_bir_loop_motif_1 = []
        for i in range(loop_length-1):
            bul_int_bir_loop_motif_1.append(motif_info_data_df.at[base_pair[0]+i, 'motif_1'])
    
        if bul_int_bir_loop_motif_1.count("hairpin_loop") >= 4:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bifurcation_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bifurcation_loop"
        else:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bul_int_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bul_int_loop"
    '''
    # 5.2.1 distinguish bifurcation loop from bulge/interior loop from number of base pairs point of view
    
    for base_pair in temp_base_pair_index:
        base_pairs_bul_int_bir_loop = []
        base_pairs_bul_int_bir_loop.append(base_pair)
        
        loop_length = base_pair[1]-base_pair[0]
        for i in range(loop_length-1):
            if motif_info_data_df.at[base_pair[0]+i, 'structure'] == "(":
                base_pair_star = motif_info_data_df.at[base_pair[0]+i, 'base_pair_index']
                break
        
        base_pairs_bul_int_bir_loop.append(base_pair_star)
        
        structure_indicator = base_pair_star[1]
        
        while False in motif_info_data_df.loc[structure_indicator:base_pair[1]-2, ['base_pair_index']].isnull()['base_pair_index'].unique():
            index_base_pair_star_new = motif_info_data_df.loc[structure_indicator:base_pair[1]-2, ['base_pair_index']].first_valid_index()
            base_pair_star_new = motif_info_data_df.at[index_base_pair_star_new, 'base_pair_index']
            
            base_pairs_bul_int_bir_loop.append(base_pair_star_new)
            structure_indicator = base_pair_star_new[1]
        
        base_pairs_bul_int_bir_loop.sort(key = lambda pair: pair[0])
        #print("The base pairs in this bifurcation/bulge/interior loop is ", base_pairs_bul_int_bir_loop)
        
        motif_info_data_df.at[base_pair[0]-1, 'motif_1_bul_int_bir_loop_base_pairs'] = base_pairs_bul_int_bir_loop
        motif_info_data_df.at[base_pair[1]-1, 'motif_1_bul_int_bir_loop_base_pairs'] = base_pairs_bul_int_bir_loop
        
        if len(motif_info_data_df.at[base_pair[0]-1, 'motif_1_bul_int_bir_loop_base_pairs'])==2:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bul_int_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bul_int_loop"
        elif len(motif_info_data_df.at[base_pair[0]-1, 'motif_1_bul_int_bir_loop_base_pairs'])>=3:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bifurcation_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bifurcation_loop"
            
        
    # 6. Distinguish the interior loop from bulge loop
    
    motif_info_data_df['motif_1_bul_int_edge_len']=0
    motif_info_data_df['motif_1_bul_int_edge_len'] = motif_info_data_df['motif_1_bul_int_edge_len'].astype(int)
    
    
    # 6.1 obtain the index of base pairs whose motif_1 equals to bul_int_loop 
    motif_info_data_df_temp = motif_info_data_df[motif_info_data_df["motif_1"]=="bul_int_loop"].reset_index(drop=True)
    
    num_rows_motif_info_data_df_temp = motif_info_data_df_temp.shape[0]
    
    temp_base_pair_index = []
    
    for i in range(num_rows_motif_info_data_df_temp):
        if motif_info_data_df_temp.at[i, 'base_pair_index'] not in temp_base_pair_index:
            temp_base_pair_index.append(motif_info_data_df_temp.at[i, 'base_pair_index'])
         
            
    # 6.2 distinguish the interior loop from bulge loop
    
    for base_pair in temp_base_pair_index:
        # 6.2.1 distinguish the interior loop from bulge loop and obtain the length of the edge of the bulge/interior loop
        loop_length = base_pair[1]-base_pair[0]
        for i in range(loop_length-1):
            if motif_info_data_df.at[base_pair[0]+i, 'structure'] == "(":
                base_pair_2 = motif_info_data_df.at[base_pair[0]+i, 'base_pair_index']
                break
        
        if base_pair_2[0] - 1 == base_pair[0] or base_pair[1] - 1 == base_pair_2[1]:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bulge_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bulge_loop"
        else:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "interior_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "interior_loop"
        
        motif_info_data_df.at[base_pair[0]-1, 'motif_1_bul_int_edge_len'] = base_pair_2[0]-base_pair[0]+1
        motif_info_data_df.at[base_pair[1]-1, 'motif_1_bul_int_edge_len'] = motif_info_data_df.at[base_pair[0]-1, 'motif_1_bul_int_edge_len']
        
        # 6.2.2 determine the sequence of interior loop and bulge loop
        loop_length_1 = base_pair_2[0]-base_pair[0]
        loop_length_2 = base_pair[1]-base_pair_2[1]
        motif_1_sequence = motif_info_data_df.at[base_pair[0]-1, 'nt']
        
        for i in range(loop_length_1):
            motif_1_sequence += motif_info_data_df.at[base_pair[0]+i, 'nt']
        
        motif_1_sequence += motif_info_data_df.at[base_pair_2[1]-1, 'nt']
        
        for i in range(loop_length_2):
            motif_1_sequence += motif_info_data_df.at[base_pair_2[1]+i, 'nt']
        
        motif_info_data_df.at[base_pair[0]-1, 'motif_1_seq'] = motif_1_sequence
        motif_info_data_df.at[base_pair[1]-1, 'motif_1_seq'] = motif_1_sequence
        
        # 6.2.3 determine the index of interior loop and bulge loop
        loop_index = []
        for i in range(loop_length_1 + 1):
            loop_index.append(motif_info_data_df.at[base_pair[0]-1+i, 'index'])
        
        for i in range(loop_length_2 + 1):
            loop_index.append(motif_info_data_df.at[base_pair_2[1]-1+i, 'index'])
        
        motif_info_data_df.at[base_pair[0]-1, 'motif_1_index'] = loop_index
        motif_info_data_df.at[base_pair[1]-1, 'motif_1_index'] = motif_info_data_df.at[base_pair[0]-1, 'motif_1_index']
        
    '''    
    # 7. Obtain the motif's info of unpaired "."
    
    temp_motif_index = []
    
    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1_index'] not in temp_motif_index and\
            not np.isnan(motif_info_data_df.at[i, 'motif_1_index']).any() and\
                (motif_info_data_df.at[i, 'motif_1'] == "hairpin_loop" or motif_info_data_df.at[i, 'motif_1'] == "interior_loop" or\
                 motif_info_data_df.at[i, 'motif_1'] == "bulge_loop"):
                    temp_motif_index.append(motif_info_data_df.at[i, 'motif_1_index'])
            
                    for j in motif_info_data_df.at[i, 'motif_1_index']:
                        if motif_info_data_df.at[j-1, 'structure'] == ".":
                            motif_info_data_df.at[j-1, 'motif_1'] = motif_info_data_df.at[i, 'motif_1']
                            motif_info_data_df.at[j-1, 'motif_1_index'] = motif_info_data_df.at[i, 'motif_1_index']
                            motif_info_data_df.at[j-1, 'motif_1_seq'] = motif_info_data_df.at[i, 'motif_1_seq']
                            motif_info_data_df.at[j-1, 'motif_1_bul_int_edge_len'] = motif_info_data_df.at[i, 'motif_1_bul_int_edge_len']
    '''                
                
    # 8. Obtain the sequence and motif index of bifurcation loop in motif 1
    
    # 8.1 obtain the index of base pairs whose motif_1 equals to bifurcation_loop
     
    motif_info_data_df_temp = motif_info_data_df[motif_info_data_df["motif_1"]=="bifurcation_loop"].reset_index(drop=True)
    
    num_rows_motif_info_data_df_temp = motif_info_data_df_temp.shape[0]
    
    temp_base_pair_index = []
    
    for i in range(num_rows_motif_info_data_df_temp):
        if motif_info_data_df_temp.at[i, 'base_pair_index'] not in temp_base_pair_index:
            temp_base_pair_index.append(motif_info_data_df_temp.at[i, 'base_pair_index'])
    
            
    # 8.2 determine the motif index and sequence of bifurcation loop
    for base_pair in temp_base_pair_index:
        
        # 8.2.1 obtain the motif index of bifurcation loop
        '''
        base_pairs_bifurcation_loop = []
        base_pairs_bifurcation_loop.append(base_pair)
        loop_index = []
        loop_index.append(motif_info_data_df.at[base_pair[0]-1, 'index'])
        loop_index.append(motif_info_data_df.at[base_pair[1]-1, 'index'])
        
        loop_length = base_pair[1]-base_pair[0]
        for i in range(loop_length-1):
            if motif_info_data_df.at[base_pair[0]+i, 'structure'] == "(":
                base_pair_star = motif_info_data_df.at[base_pair[0]+i, 'base_pair_index']
                break
        
        base_pairs_bifurcation_loop.append(base_pair_star)
        loop_index.append(motif_info_data_df.at[base_pair_star[0]-1, 'index'])
        loop_index.append(motif_info_data_df.at[base_pair_star[1]-1, 'index'])
        
        structure_indicator = base_pair_star[1]
        
        while False in motif_info_data_df.loc[structure_indicator:base_pair[1]-2, ['base_pair_index']].isnull()['base_pair_index'].unique():
            index_base_pair_star_new = motif_info_data_df.loc[structure_indicator:base_pair[1]-2, ['base_pair_index']].first_valid_index()
            base_pair_star_new = motif_info_data_df.at[index_base_pair_star_new, 'base_pair_index']
            
            loop_index.append(motif_info_data_df.at[base_pair_star_new[0]-1, 'index'])
            loop_index.append(motif_info_data_df.at[base_pair_star_new[1]-1, 'index'])
            
            base_pairs_bifurcation_loop.append(base_pair_star_new)
            structure_indicator = base_pair_star_new[1]
        
        base_pairs_bifurcation_loop.sort(key = lambda pair: pair[0])
        #print("The base pairs in this bifurcation loop is ", base_pairs_bifurcation_loop)
        '''
        base_pairs_bifurcation_loop = motif_info_data_df.at[base_pair[0]-1, 'motif_1_bul_int_bir_loop_base_pairs']
        #print("The base pairs in this bifurcation loop is ", base_pairs_bifurcation_loop)
        
        loop_index = [each_nt for each_base_pair in base_pairs_bifurcation_loop for each_nt in each_base_pair]
        
        edge_length = base_pairs_bifurcation_loop[1][0]-base_pairs_bifurcation_loop[0][0]
        #print("The first edge length is ", edge_length)
        for i in range(edge_length-1):
            if motif_info_data_df.at[base_pairs_bifurcation_loop[0][0]+i, 'structure'] == ".":
                loop_index.append(motif_info_data_df.at[base_pairs_bifurcation_loop[0][0]+i, 'index'])
            
        edge_length = base_pairs_bifurcation_loop[0][1]-base_pairs_bifurcation_loop[-1][1]
        #print("The second edge length is ", edge_length)
        for i in range(edge_length-1):
            if motif_info_data_df.at[base_pairs_bifurcation_loop[-1][1]+i, 'structure'] == ".":
                loop_index.append(motif_info_data_df.at[base_pairs_bifurcation_loop[-1][1]+i, 'index'])
        
        base_pairs_bifurcation_loop.pop(0)
        
        length_base_pairs_bifurcation_loop = len(base_pairs_bifurcation_loop)
        
        for i in range(length_base_pairs_bifurcation_loop-1):
            edge_length = base_pairs_bifurcation_loop[i+1][0]-base_pairs_bifurcation_loop[i][1]
            for j in range(edge_length-1):
                if motif_info_data_df.at[base_pairs_bifurcation_loop[i][1]+j, 'structure'] == ".":
                    loop_index.append(motif_info_data_df.at[base_pairs_bifurcation_loop[i][1]+j, 'index'])
            
        loop_index = list(map(np.int64, loop_index))
        loop_index.sort()
        #print("The index of nucleotides in this bifurcation loop is ", loop_index)
        
        motif_info_data_df.at[base_pair[0]-1, 'motif_1_index'] = loop_index
        motif_info_data_df.at[base_pair[1]-1, 'motif_1_index'] = motif_info_data_df.at[base_pair[0]-1, 'motif_1_index']
    
        # 8.2.2 obtain the sequence of bifurcation loop
        len_bifurcation_loop = len(motif_info_data_df.at[base_pair[0]-1, 'motif_1_index'])
        motif_1_sequence = motif_info_data_df.at[base_pair[0]-1, 'nt']
        for i in range(1, len_bifurcation_loop):
            motif_1_sequence += motif_info_data_df.at[motif_info_data_df.at[base_pair[0]-1, 'motif_1_index'][i]-1, 'nt']
            
        
        motif_info_data_df.at[base_pair[0]-1, 'motif_1_seq'] = motif_1_sequence
        motif_info_data_df.at[base_pair[1]-1, 'motif_1_seq'] = motif_1_sequence
        
          
    # 7. Obtain the motif's info of unpaired "."
    
    temp_motif_index = []
    
    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1_index'] not in temp_motif_index and\
            not np.isnan(motif_info_data_df.at[i, 'motif_1_index']).any() and\
                (motif_info_data_df.at[i, 'motif_1'] == "hairpin_loop" or motif_info_data_df.at[i, 'motif_1'] == "interior_loop" or\
                 motif_info_data_df.at[i, 'motif_1'] == "bulge_loop" or motif_info_data_df.at[i, 'motif_1'] == "bifurcation_loop"):
                    temp_motif_index.append(motif_info_data_df.at[i, 'motif_1_index'])
            
                    for j in motif_info_data_df.at[i, 'motif_1_index']:
                        if motif_info_data_df.at[j-1, 'structure'] == ".":
                            motif_info_data_df.at[j-1, 'motif_1'] = motif_info_data_df.at[i, 'motif_1']
                            motif_info_data_df.at[j-1, 'motif_1_index'] = motif_info_data_df.at[i, 'motif_1_index']
                            motif_info_data_df.at[j-1, 'motif_1_seq'] = motif_info_data_df.at[i, 'motif_1_seq']
                            motif_info_data_df.at[j-1, 'motif_1_bul_int_edge_len'] = motif_info_data_df.at[i, 'motif_1_bul_int_edge_len']
                            motif_info_data_df.at[j-1, 'motif_1_bul_int_bir_loop_base_pairs'] = motif_info_data_df.at[i, 'motif_1_bul_int_bir_loop_base_pairs']
    
            
    # 9. Obtain the motif 2 info of each nucleotide 
    
    
    motif_info_data_df['motif_2']=np.nan
    motif_info_data_df['motif_2'] = motif_info_data_df['motif_2'].astype(object)
    
    motif_info_data_df['motif_2_seq']=np.nan
    motif_info_data_df['motif_2_seq'] = motif_info_data_df['motif_2_seq'].astype(object)
    
    motif_info_data_df['motif_2_index']=np.nan
    motif_info_data_df['motif_2_index'] = motif_info_data_df['motif_2_index'].astype(object) 
    
    motif_info_data_df['motif_2_bul_int_edge_len']=0
    motif_info_data_df['motif_2_bul_int_edge_len'] = motif_info_data_df['motif_2_bul_int_edge_len'].astype(int)
    
    
    temp_motif_index = []
    
    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1_index'] not in temp_motif_index and\
            not np.isnan(motif_info_data_df.at[i, 'motif_1_index']).any():
                for j in motif_info_data_df.at[i, 'motif_1_index']:
                    if motif_info_data_df.at[j-1, 'motif_1_index'] != motif_info_data_df.at[i, 'motif_1_index']:
                        motif_info_data_df.at[j-1, 'motif_2'] = motif_info_data_df.at[i, 'motif_1']
                        motif_info_data_df.at[j-1, 'motif_2_seq'] = motif_info_data_df.at[i, 'motif_1_seq']
                        motif_info_data_df.at[j-1, 'motif_2_index'] = motif_info_data_df.at[i, 'motif_1_index']
                        motif_info_data_df.at[j-1, 'motif_2_bul_int_edge_len'] = motif_info_data_df.at[i, 'motif_1_bul_int_edge_len']
                        
                temp_motif_index.append(motif_info_data_df.at[i, 'motif_1_index'])
                
                     
    # 10. Calculate the FE of each motif
    
    motif_info_data_df['motif_1_FE']=np.nan
    motif_info_data_df['motif_1_FE'] = motif_info_data_df['motif_1_FE'].astype(float)
    
    motif_info_data_df['motif_2_FE']=np.nan
    motif_info_data_df['motif_2_FE'] = motif_info_data_df['motif_2_FE'].astype(float)
    
       
    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1'] == "stack":
            motif_info_data_df.at[i, 'motif_1_FE'] = RNA.fold_compound(motif_info_data_df.at[i, 'motif_1_seq']).eval_int_loop(i=1,j=4, k=2, l=3)
        elif motif_info_data_df.at[i, 'motif_1'] == "hairpin_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_1_seq'])
            motif_info_data_df.at[i, 'motif_1_FE'] = RNA.fold_compound(motif_info_data_df.at[i, 'motif_1_seq']).eval_hp_loop(i=1,j=loop_len)
        elif motif_info_data_df.at[i, 'motif_1'] == "bulge_loop" or motif_info_data_df.at[i, 'motif_1'] == "interior_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_1_seq'])
            motif_info_data_df.at[i, 'motif_1_FE'] =\
                RNA.fold_compound(motif_info_data_df.at[i, 'motif_1_seq']).eval_int_loop(i=1,j=loop_len, k=motif_info_data_df.at[i, 'motif_1_bul_int_edge_len'].item(), l=motif_info_data_df.at[i, 'motif_1_bul_int_edge_len'].item()+1)
        elif motif_info_data_df.at[i, 'motif_1'] == "bifurcation_loop":
            motif_info_data_df.at[i, 'motif_1_FE'] = RNA.fold_compound(RNA_seq).eval_loop_pt(i=motif_info_data_df.at[i, 'motif_1_index'][0].item(), pt = RNA.ptable(RNA_ss))
        else:
            motif_info_data_df.at[i, 'motif_1_FE'] = 0
        
        if motif_info_data_df.at[i, 'motif_2'] == "stack":
            motif_info_data_df.at[i, 'motif_2_FE'] = RNA.fold_compound(motif_info_data_df.at[i, 'motif_2_seq']).eval_int_loop(i=1,j=4, k=2, l=3)
        elif motif_info_data_df.at[i, 'motif_2'] == "hairpin_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_2_seq'])
            motif_info_data_df.at[i, 'motif_2_FE'] = RNA.fold_compound(motif_info_data_df.at[i, 'motif_2_seq']).eval_hp_loop(i=1,j=loop_len)
        elif motif_info_data_df.at[i, 'motif_2'] == "bulge_loop" or motif_info_data_df.at[i, 'motif_2'] == "interior_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_2_seq'])
            motif_info_data_df.at[i, 'motif_2_FE'] =\
                RNA.fold_compound(motif_info_data_df.at[i, 'motif_2_seq']).eval_int_loop(i=1,j=loop_len, k=motif_info_data_df.at[i, 'motif_2_bul_int_edge_len'].item(), l=motif_info_data_df.at[i, 'motif_2_bul_int_edge_len'].item()+1)
        elif motif_info_data_df.at[i, 'motif_2'] == "bifurcation_loop":
            motif_info_data_df.at[i, 'motif_2_FE'] = RNA.fold_compound(RNA_seq).eval_loop_pt(i=motif_info_data_df.at[i, 'motif_2_index'][0].item(), pt = RNA.ptable(RNA_ss))
        else:
            motif_info_data_df.at[i, 'motif_2_FE'] = 0
            
            
    
    del motif_info_data_df["motif_1_bul_int_edge_len"]
    del motif_info_data_df["motif_2_bul_int_edge_len"]
    del motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs']
    return motif_info_data_df
    



'''
# Test
RNA_seq = "CUCGUCUAGUCAUUUCUGGCCCCACUGGAGGUCGAG"
RNA_ss = "((((.((((......))))((((...)).)).))))"
result = RNA_motif_info_extraction(RNA_seq, RNA_ss)
#result.to_csv("motif_info_results_test.csv",index=False)
'''













