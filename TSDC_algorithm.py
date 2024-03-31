from math import pi, cos, sin
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.spatial.distance import cosine

def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u

def cosine_new(u, v):
    """
    computes the cosine similarity between 1-D arrays.
   
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    u_1=np.linalg.norm(u)
    v_1=np.linalg.norm(v)
    if u_1*v_1==0:
        dist=0
    else:
        dist =np.abs(1-cosine(u,v))
    return dist

def Row_Cosine_Center(block_1): 
    """
    computes community center of out-communities
   
    """
    block_1_normalized=normalize(block_1,axis=1)#对每行做标准化
    block_1_normalized_mean=np.mean(block_1_normalized,axis=0)#求各个行的平均
    # if np.linalg.norm(block_1_normalized_mean,ord=1)==0:
    #     print(block_1_normalized_mean)
    #     # block_11_normalized_mean=np.full(len(block_1_normalized_mean),1/len(block_1_normalized_mean))
    #     block_11_normalized_mean=block_1_normalized_mean*len(block_1_normalized_mean)
    # else:
    block_11_normalized_mean=block_1_normalized_mean/(np.linalg.norm(block_1_normalized_mean,ord=1)+1e-20)*len(block_1_normalized_mean)
    
    return block_11_normalized_mean

def Col_Cosine_Center(block_2): 
    """
    computes community center of in-communities
   
    """
    block_2_normalized=normalize(block_2,axis=0)
    block_2_normalized_mean=np.mean(block_2_normalized,axis=1)
    # if np.linalg.norm(block_2_normalized_mean,ord=1)==0:
    #     print(block_2_normalized_mean)
    #     # block_22_normalized_mean=np.full(len(block_2_normalized_mean),1/len(block_2_normalized_mean))
    #     block_22_normalized_mean=block_2_normalized_mean*len(block_2_normalized_mean)
    # else:
    block_22_normalized_mean=block_2_normalized_mean/(np.linalg.norm(block_2_normalized_mean,ord=1)+1e-20)*len(block_2_normalized_mean)
    return block_22_normalized_mean

def TSDC_algorithm(data_1,C,Z,K_1,K_2,n,m,arr_vector_1,arr_vector_2,steps=5000,tolerance=0.001):
    '''
    data_1:data matrix
    C:vector of out-community assignment
    Z:vector of in-community assignment
    K_1: the number of out-communities
    K_2: the number of in-communities
    n: the number of out-nodes
    m: the number of in-nodes
    arr_vector_1: ground truth of out-communities
    arr_vector_2: ground truth of in-communities
    steps: iterations
    tolerance: relative difference of the objective function
    
    '''
    NMI_summary=np.zeros(steps)
    for step in range(steps):
        
        ########################################################################################### initialize community center of out-communities,in-communities
        mu=np.zeros(shape=(K_1,m))
        mu_tilde=np.zeros(shape=(n,K_2))
        for  k_1 in range(0,K_1):
            for k_2 in range(0,K_2):
                row_ind= np.asarray(np.where(C==k_1))[0]
                col_ind= np.asarray(np.where(Z==k_2))[0]                
                block_A=data_1[:,col_ind][row_ind,:]
                if (len(row_ind)*len(col_ind)==0):
                    row_center=np.zeros(len(col_ind))
                else:
                    row_center=Row_Cosine_Center(block_A)
                if (len(row_ind)*len(col_ind)==0):
                    col_center=np.zeros(len(row_ind))
                else:
                    col_center=Col_Cosine_Center(block_A)
                mu[k_1,col_ind]=row_center
                mu_tilde[row_ind,k_2]=col_center
        
        ########################################################################################### given community centers and Z, update C
        for i in range(0,n):
            local_score_row=np.zeros(K_1)
            for p in range(0,K_1):
                SumCosine=0
                for q in range(0,K_2):
                    col_inde=np.asarray(np.where(Z==q))[0] 
                    SumCosine=SumCosine+cosine_new(data_1[i,col_inde],mu[p,col_inde])/K_2
                local_score_row[p]= SumCosine
            C[i]=int(np.array(np.where(local_score_row==np.max(local_score_row)))[0][0])
            # print(local_score_row)
            
        ########################################################################################### given community centers and C, update Z
        for j in range(0,m):
            local_score_col=np.zeros(K_2)
            for qq in range(0,K_2):
                SumCosine_1=0
                for ppp in range(0,K_1):
                    row_index=np.asarray(np.where(C==ppp))[0] 
                    SumCosine_1=SumCosine_1+cosine_new(data_1[row_index,j],mu_tilde[row_index,qq])/K_1
                local_score_col[qq]=SumCosine_1
            Z[j]=int(np.array(np.where(local_score_col==np.max(local_score_col)))[0][0])

        NMI_dividedcos_test_1=normalized_mutual_info_score(arr_vector_1,C)
        NMI_dividedcos_test_2=normalized_mutual_info_score(arr_vector_2,Z)
        NMI_summary[step]=(NMI_dividedcos_test_1+NMI_dividedcos_test_2)/2
        if step>=1:
            e=np.abs(NMI_summary[step]-NMI_summary[step-1])
            if e == 0:
                break 
    return C,Z 
