from scipy import linalg
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score


def rank_one_approximation_error(block_1):
    U, s, VT = np.linalg.svd(block_1)
    U_test=np.asmatrix(U[:,0]).T
    V_test=np.asmatrix(VT[0,:])
    rank1_test=s[0]*np.dot(U_test,V_test)
    rank1_error=np.power(np.linalg.norm(block_1-rank1_test),2)
    return rank1_error

def DOM_algorithm(data_1,C,Z,K_1,K_2,n,m,arr_vector_1,arr_vector_2,steps=5000,tolerance=0.001):
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
        
        for i in range(0,n):
            local_loss_row=np.zeros(K_1)
            for p in range(0,K_1):
                C[i]=p
                row_inde=np.asarray(np.where(C==p))[0] 
                row_inde_diff=np.array(list(set(row_inde).difference({i})))    
                SumRankLoss=0
                for q in range(0,K_2):
                    col_inde=np.asarray(np.where(Z==q))[0] 
                    if len(col_inde)==0:
                        SumRankLoss=SumRankLoss+0
                    elif len(row_inde_diff)==0:
                        block_row=data_1[:,col_inde][row_inde,:]
                        SumRankLoss=SumRankLoss+rank_one_approximation_error(block_row)
                    else:
                        block_row=data_1[:,col_inde][row_inde,:]
                        block_row_diff=data_1[:,col_inde][row_inde_diff,:]
                        SumRankLoss=SumRankLoss+rank_one_approximation_error(block_row)-rank_one_approximation_error(block_row_diff)
                local_loss_row[p]= SumRankLoss
            C[i]=int(np.array(np.where(local_loss_row==np.min(local_loss_row)))[0][0])    ###### given Z,update C
        
        for j in range(0,m):
            local_loss_col=np.zeros(K_2)
            for qq in range(0,K_2):
                Z[j]=qq
                col_index=np.asarray(np.where(Z==qq))[0]
                col_index_diff=np.array(list(set(col_index).difference({j}))) 
                SumRankLoss_1=0
                for ppp in range(0,K_1):
                    row_index=np.asarray(np.where(C==ppp))[0] 
                    if len(row_index)==0:
                        SumRankLoss_1=SumRankLoss_1+0
                    elif len(col_index_diff)==0:
                        block_col=data_1[:,col_index][row_index,:]
                        SumRankLoss_1=SumRankLoss_1+rank_one_approximation_error(block_col)
                    else:
                        block_col=data_1[:,col_index][row_index,:]
                        block_col_diff=data_1[:,col_index_diff][row_index,:]
                        SumRankLoss_1=SumRankLoss_1+rank_one_approximation_error(block_col)-rank_one_approximation_error(block_col_diff)
                local_loss_col[qq]=SumRankLoss_1
            Z[j]=int(np.array(np.where(local_loss_col==np.min(local_loss_col)))[0][0])     ###### given C,update Z

        NMI_dividedcos_test_1=normalized_mutual_info_score(arr_vector_1,C)
        NMI_dividedcos_test_2=normalized_mutual_info_score(arr_vector_2,Z)
        NMI_summary[step]=(NMI_dividedcos_test_1+NMI_dividedcos_test_2)/2
        if step>=1:
            e=np.abs(NMI_summary[step]-NMI_summary[step-1])
            if e == 0:
                break 
    return C,Z  


