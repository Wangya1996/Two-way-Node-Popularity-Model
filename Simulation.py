import numpy as np
from scipy.stats import bernoulli
from random import random

def clu_vec_to_mat(vec_1):
    mat_1=np.zeros(shape=(len(vec_1),np.max(vec_1)+1))
    for _ in range(0,len(vec_1)):
        mat_1[_,vec_1[_]]=1
    return mat_1


def simulation_data_mat(K_1,K_2,n,m,sigma,method):
    '''
    K_1: the number of out-communities
    K_2: the number of in-communities
    n: the number of out-nodes
    m: the number of in-nodes
    sigma: the variance of normal distribution
    method: 'Normal','Bernoulli', and 'Poisson', the data generating machanism
    
    '''
    ground_truth_cluster_label_1=np.random.randint(0,K_1,size=n)
    ground_truth_cluster_label_2=np.random.randint(0,K_2,size=m)
    C_matrix=clu_vec_to_mat(ground_truth_cluster_label_1)
    Z_matrix=clu_vec_to_mat(ground_truth_cluster_label_2)
    
    if method=='Normal': ###### generate data matrix from Normal distribution
        lamda_1=np.random.uniform(0,1,n*K_2).reshape(n,K_2)
        lamda_2=np.random.uniform(0,1,m*K_1).reshape(m,K_1)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=np.random.normal(P[i,j],sigma)
    
    elif method=='Bernoulli': ###### generate data matrix from Bernoulli distribution
        lamda_1=np.random.uniform(0,1,n*K_2).reshape(n,K_2)
        lamda_2=np.random.uniform(0,1,m*K_1).reshape(m,K_1)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=bernoulli.rvs(size=1,p=P[i,j])
                
    else:                     ###### generate data matrix from Poisson distribution
        lamda_1=np.random.uniform(1,3,n*K_2).reshape(n,K_2)
        lamda_2=np.random.uniform(1,3,m*K_1).reshape(m,K_1)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=np.random.poisson(lam=P[i,j])
    return A,ground_truth_cluster_label_1,ground_truth_cluster_label_2


def off_diagonal_blocks_zeros(arr,ratio,block_label):
    n,K=arr.shape
    num_elements = int(arr.size * ratio)
    flat_array = arr.flatten()
    min_indices = np.argpartition(flat_array, num_elements)[:num_elements]
    flat_array[min_indices] = 0
    modified_array = flat_array.reshape(arr.shape)
    for k in range(K):
        modified_array[np.where(block_label==k),k]=arr[np.where(block_label==k),k]
    return modified_array



def simulation_data_mat_sparsity(K_1,K_2,n,m,sigma,spar_level,method):
    '''
    K_1: the number of out-communities
    K_2: the number of in-communities
    n: the number of out-nodes
    m: the number of in-nodes
    sigma: the variance of normal distribution
    spar_level: control the level of sparsity
    method: 'Normal','Bernoulli', and 'Poisson', the data generating machanism
    
    '''
    random_numbers=np.random.multinomial(1, np.full(K_1, 1/K_1), size=n)
    ground_truth_cluster_label_1=np.array(np.where(random_numbers==1)[1])#等比例生成cluster
    random_numbers=np.random.multinomial(1, np.full(K_2, 1/K_2), size=m)
    ground_truth_cluster_label_2=np.array(np.where(random_numbers==1)[1])
    
    if method=='Normal': ###### generate data matrix from Normal distribution
        lamda_11=np.random.uniform(0,1,n*K_2).reshape(n,K_2)
        # print(lamda_11)
        lamda_1=off_diagonal_blocks_zeros(lamda_11,spar_level,ground_truth_cluster_label_1)
        # print(lamda_1)
        lamda_22=np.random.uniform(0,1,m*K_1).reshape(m,K_1)
        lamda_2=off_diagonal_blocks_zeros(lamda_22,spar_level,ground_truth_cluster_label_2)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=np.random.normal(P[i,j],sigma)
    
    elif method=='Bernoulli': ###### generate data matrix from Bernoulli distribution
        lamda_11=np.random.uniform(0,1,n*K_2).reshape(n,K_2)
        lamda_1=off_diagonal_blocks_zeros(lamda_11,spar_level,ground_truth_cluster_label_1)
        lamda_22=np.random.uniform(0,1,m*K_1).reshape(m,K_1)
        lamda_2=off_diagonal_blocks_zeros(lamda_22,spar_level,ground_truth_cluster_label_2)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=bernoulli.rvs(size=1,p=P[i,j])
                
    else:                     ###### generate data matrix from Poisson distribution
        lamda_11=np.random.uniform(1,3,n*K_2).reshape(n,K_2)
        lamda_1=off_diagonal_blocks_zeros(lamda_11,spar_level,ground_truth_cluster_label_1)
        lamda_22=np.random.uniform(1,3,m*K_1).reshape(m,K_1)
        lamda_2=off_diagonal_blocks_zeros(lamda_22,spar_level,ground_truth_cluster_label_2)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=np.random.poisson(lam=P[i,j])
    return A,ground_truth_cluster_label_1,ground_truth_cluster_label_2

#20240313 sub-Gaussian
def simulation_data_mat_sub_Gaussian(K_1,K_2,n,m,sigma,method):
    '''
    K_1: the number of out-communities
    K_2: the number of in-communities
    n: the number of out-nodes
    m: the number of in-nodes
    sigma: the variance of normal distribution
    method: 'Normal','Bernoulli', and 'Poisson', the data generating machanism
    
    '''
    ground_truth_cluster_label_1=np.random.randint(0,K_1,size=n)
    ground_truth_cluster_label_2=np.random.randint(0,K_2,size=m)
    C_matrix=clu_vec_to_mat(ground_truth_cluster_label_1)
    Z_matrix=clu_vec_to_mat(ground_truth_cluster_label_2)
    
    if method=='Normal': ###### generate data matrix from Normal distribution
        lamda_1=np.random.uniform(0,1,n*K_2).reshape(n,K_2)
        lamda_2=np.random.uniform(0,1,m*K_1).reshape(m,K_1)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=np.random.normal(P[i,j],sigma)
    
    elif method=='Bernoulli': ###### generate data matrix from Bernoulli distribution
        lamda_1=np.random.uniform(0,1,n*K_2).reshape(n,K_2)
        lamda_2=np.random.uniform(0,1,m*K_1).reshape(m,K_1)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=bernoulli.rvs(size=1,p=P[i,j])
                
    else:                     ###### generate data matrix from Poisson distribution
        lamda_1=np.random.uniform(1,3,n*K_2).reshape(n,K_2)
        lamda_2=np.random.uniform(1,3,m*K_1).reshape(m,K_1)
        P=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                P[i,j]=lamda_1[i,ground_truth_cluster_label_2[j]]*lamda_2[j,ground_truth_cluster_label_1[i]]
        A=np.zeros(shape=(n,m))
        for i in range(0,n):
            for j in range(0,m):
                A[i,j]=np.random.poisson(lam=P[i,j])
    return A,ground_truth_cluster_label_1,ground_truth_cluster_label_2