import ROA_algorithm as ROA_algorithm
import TSDC_algorithm as TSDC_algorithm
import Competitive_algorithms as Competitive_algorithms
import Simulation as Simulation
from sklearn.metrics.cluster import normalized_mutual_info_score
import math
import numpy as np

def rank_one_approximation_error(block_1):
    U, s, VT = np.linalg.svd(block_1)
    U_test=np.asmatrix(U[:,0]).T
    V_test=np.asmatrix(VT[0,:])
    rank1_test=s[0]*np.dot(U_test,V_test)
    rank1_error=np.power(np.linalg.norm(block_1-rank1_test),2)
    return rank1_error

def comm_num_est(array,shift):
    min_index_flat =np.argmin(array)
    min_index_2d = np.unravel_index(min_index_flat, array.shape)
    new_index_2d = (min_index_2d[0] + shift[0], min_index_2d[1] + shift[1])
    return new_index_2d

def num_cluster_obj(label_row,label_col,data_mat,K_hat,L_hat):
    '''
    label_row: vector of out-community assignment
    label_col: vector of in-community assignment
    K_hat: the number of out-communities
    L_hat: the number of in-communities

    return the penalized objective function
    
    '''
    RankLoss=0
    for k1 in range(K_hat):
        for k2 in range(L_hat):
            row_index=np.asarray(np.where(label_row==k1))[0]
            col_index=np.asarray(np.where(label_col==k2))[0]
            if len(col_index)*len(row_index)==0:
                RankLoss=RankLoss+0
            else:
                block=data_mat[:,col_index][row_index,:]
                RankLoss=RankLoss+rank_one_approximation_error(block)
    pen_1=data_mat.shape[0]*L_hat*np.sqrt(math.log(data_mat.shape[0])*(math.log(L_hat)**3))
    pen_2=data_mat.shape[1]*K_hat*np.sqrt(math.log(data_mat.shape[1])*(math.log(K_hat)**3))
    rho_A=np.sum(np.abs(data_mat))/(np.prod(data_mat.size))    
    return RankLoss+0.5*rho_A*(pen_1+pen_2)