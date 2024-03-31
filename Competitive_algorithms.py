from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from scipy import linalg
from sklearn.preprocessing import normalize
################################################### OMPSC algorithm

def Spectral_Clustering(array_2,cluster_number):
    sample_size=array_2.shape[0]
    array_21=np.linalg.inv(np.sqrt(np.diag(np.sum(array_2,axis=1))))
    array_23=np.diag(np.sum(array_2,axis=1))
    Laplacian_mat=np.dot(np.dot(array_21,(array_23-array_2)),array_21)
    e_vals,e_vecs = np.linalg.eig(Laplacian_mat)
    sorted_indices = np.argsort(e_vals)
    U_test_sc=e_vecs[:,sorted_indices[:-cluster_number-1:-1]].real.astype(np.float32)
    T_test_sc=normalize(U_test_sc,axis=1)
    clustering_results_new_sc = KMeans(n_clusters=cluster_number, random_state=1,init='k-means++',n_init=20).fit(T_test_sc)
    return clustering_results_new_sc.labels_

def orthogonal_mp(X, y, n_nonzero_coefs, eps=None):
    residual = y
    idx = []
    if  eps == None:
        stopping_condition = lambda: len(idx) == n_nonzero_coefs 
    else:
        stopping_condition = lambda: np.inner(residual, residual) <= eps
    while not stopping_condition():
        lam = np.abs(np.dot(residual, X)).argmax()
        idx.append(lam)
        gamma, _, _, _ = linalg.lstsq(X[:, idx], y)
        residual = y - np.dot(X[:, idx], gamma)
    return gamma, idx

def OMPSC_algorithm(data_1,K_1,K_2,n,m):
    '''
    data_1:data matrix
    C:vector of out-community assignment
    Z:vector of in-community assignment
    K_1: the number of out-communities
    K_2: the number of in-communities
    n: the number of out-nodes
    m: the number of in-nodes
    
    '''
    W1_hat=np.zeros(shape=(n,n))
    for j in range(0,n,1):
        omp=orthogonal_mp(np.delete(data_1.T,j,axis=1), data_1.T[:,j], K_1, eps=None)
        weight=omp[0]
        index=np.array(omp[1])
        coef =np.zeros(n-1)
        for k in range(0,n-1):
             coef[k]=np.sum(weight[np.where(index==k)])
        coef_1=np.insert(coef,j,0,0)
        W1_hat[j,:]=coef_1

    W2_hat=np.zeros(shape=(m,m))
    for j in range(0,m,1):
        omp=orthogonal_mp(np.delete(data_1,j,axis=1), data_1[:,j], K_2, eps=None)
        weight=omp[0]
        index=np.array(omp[1])
        coef =np.zeros(m-1)
        for k in range(0,m-1):
             coef[k]=np.sum(weight[np.where(index==k)])
        coef_1=np.insert(coef,j,0,0)
        W2_hat[:,j]=coef_1
    
    S1_hat=np.abs(W1_hat)+np.abs(W1_hat.T)
    S2_hat=np.abs(W2_hat)+np.abs(W2_hat.T)
    Row_labels_OMPSC=Spectral_Clustering(S1_hat,K_1)
    Col_labels_OMPSC=Spectral_Clustering(S2_hat,K_2)
    return Row_labels_OMPSC,Col_labels_OMPSC


################################################### SVDK algorithm

def SVD_estimation(array_8,num_row,num_column):
    U, s, V_trans = np.linalg.svd(array_8)
    U_eatimate=U[:,:num_row]
    V_trans_eatimate=V_trans[:num_column,:]
    return U_eatimate,V_trans_eatimate

def SVDK_algorithm(data_1,K_1,K_2):
    '''
    data_1:data matrix
    K_1: the number of out-communities
    K_2: the number of in-communities
   
    '''
    UU,VV=SVD_estimation(data_1,K_1,K_2)
    Row_labels_SVDK_1=KMeans(n_clusters=K_1, random_state=1,init='k-means++',n_init=20).fit(UU)
    Row_labels_SVDK=Row_labels_SVDK_1.labels_
    Col_labels_SVDK_1=KMeans(n_clusters=K_2, random_state=1,init='k-means++',n_init=20).fit(VV.T)
    Col_labels_SVDK=Col_labels_SVDK_1.labels_
    return Row_labels_SVDK,Col_labels_SVDK


################################################### INSC algorithm
def INSC_algorithm(data_1,K_1,K_2,n,m):
    '''
    data_1:data matrix
    K_1: the number of out-communities
    K_2: the number of in-communities
    n: the number of out-nodes
    m: the number of in-nodes
    '''
    Inner_Product_Mat_Row=np.zeros(shape=(n,n))
    for i in range(0,n):
        for j in range(0,n):
            Inner_Product_Mat_Row[i,j]=np.dot(data_1[i,:],data_1[j,:])
    Inner_Product_Mat_Col=np.zeros(shape=(m,m))
    for i in range(0,m):
        for j in range(0,m):
            Inner_Product_Mat_Col[i,j]=np.dot(data_1[:,i],data_1[:,j]) 
    for i in range(0,Inner_Product_Mat_Row.shape[0]):
        Inner_Product_Mat_Row[i,i]=0
    for j in range(0,Inner_Product_Mat_Col.shape[0]):
        Inner_Product_Mat_Col[j,j]=0
    Row_labels_INSC=Spectral_Clustering(np.abs(Inner_Product_Mat_Row),K_1)
    Col_labels_INSC=Spectral_Clustering(np.abs(Inner_Product_Mat_Col),K_2)
    return Row_labels_INSC,Col_labels_INSC
    

################################################### COSSC algorithm
def COSSC_algorithm(data_1,K_1,K_2):
    '''
    data_1:data matrix
    K_1: the number of out-communities
    K_2: the number of in-communities
   
    '''
    Similarity_Mat_Row=cosine_similarity(np.asarray(data_1))
    Similarity_Mat_Col=cosine_similarity(np.asarray(data_1).T)
    for i in range(0,Similarity_Mat_Row.shape[0]):
        Similarity_Mat_Row[i,i]=0
    for j in range(0,Similarity_Mat_Col.shape[0]):
        Similarity_Mat_Col[j,j]=0
    Row_labels_COSSC=Spectral_Clustering(np.abs(Similarity_Mat_Row),K_1)
    Col_labels_COSSC=Spectral_Clustering(np.abs(Similarity_Mat_Col),K_2)
    return Row_labels_COSSC,Col_labels_COSSC




##############################################evaluation metric:
#clustering error:
def clustering_error(psi_hat, psi_star):
    """
    Calculate the clustering error between the estimated community assignment
    function psi_hat and the true community assignment function psi_star.

    Parameters:
    psi_hat (np.array): Estimated community assignments.
    psi_star (np.array): True community assignments.

    Returns:
    float: The clustering error.
    """
    # Ensure both inputs are numpy arrays
    psi_hat = np.array(psi_hat)
    psi_star = np.array(psi_star)
    
    # Number of elements
    n = len(psi_hat)
    
    # Initialize error count
    error_count = 0
    
    # Compare each pair of elements
    for i in range(n):
        for j in range(i+1, n):
            # Check if the estimated and true assignments agree for both elements of the pair
            if (psi_hat[i] == psi_hat[j]) != (psi_star[i] == psi_star[j]):
                error_count += 1
    
    # Calculate the clustering error
    clustering_error = 2 * error_count / (n * (n - 1))
    
    return clustering_error

#permutation matrix：
from itertools import permutations

def calculate_misclustered_nodes(Z, Z_true):
    n, K = Z.shape
    # Compute the l1 norm for all possible permutation matrices
    norms = []
    for PK in permutations(range(K)):
        P = np.eye(K)[:, list(PK)]
        norm = np.linalg.norm(Z @ P - Z_true, ord=1)
        norms.append(norm)
    # Compute the proportion of misclustered nodes
    misclustered_nodes = (2 * n) ** -1 * min(norms)

    return misclustered_nodes

