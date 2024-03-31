import DOM_algorithm as DOM_algorithm
import TSDC_algorithm as TSDC_algorithm
import Competitive_algorithms as Competitive_algorithms
import Simulation as Simulation
from sklearn.metrics.cluster import normalized_mutual_info_score
import time

n,m=[60,60]
K_1,K_2=[4,4]
sigma=0 #### [0,0.1,0.2,0.3,...,1]
A,ground_truth_cluster_label_1,ground_truth_cluster_label_2=Simulation.simulation_data_mat(K_1,K_2,n,m,sigma,method='Normal')
repeat_times=100
step_ROA=40
tolerance_ROA=0.000000001
iteration_time_TSDC=40
tolerance_TSDC=0.00000001
if __name__ == '__main__':
    NMI_res_row=[]
    NMI_res_col=[]
    Time_res=[]
    for repeat in range(repeat_times):
        print(repeat)
        Res_row=[]
        Res_col=[]
        Res_time=[]
        ###################################################################################################################################DOM algorithm results
        C,Z=Competitive_algorithms.SVDK_algorithm(A,K_1,K_2)
        start = time.process_time()
        Row_labels_Rank1Appro,Col_labels_Rank1Appro=DOM_algorithm.DOM_algorithm(A,C,Z,K_1,K_2,n,m,ground_truth_cluster_label_1,ground_truth_cluster_label_2,steps=step_ROA,tolerance=tolerance_ROA)
        end = time.process_time()
        Res_row.append(normalized_mutual_info_score(ground_truth_cluster_label_1,Row_labels_Rank1Appro))
        Res_col.append(normalized_mutual_info_score(ground_truth_cluster_label_2,Col_labels_Rank1Appro))
        Res_time.append(end - start)
        
        ###################################################################################################################################TSDC algorithm results
        C,Z=Competitive_algorithms.SVDK_algorithm(A,K_1,K_2)
        start = time.process_time()
        Row_labels_TSDC,Col_labels_TSDC=TSDC_algorithm.TSDC_algorithm(A,C,Z,K_1,K_2,n,m,ground_truth_cluster_label_1,ground_truth_cluster_label_2,steps=iteration_time_TSDC,tolerance=tolerance_TSDC)
        end = time.process_time()
        Res_row.append(normalized_mutual_info_score(ground_truth_cluster_label_1,Row_labels_TSDC))
        Res_col.append(normalized_mutual_info_score(ground_truth_cluster_label_2,Col_labels_TSDC))
        Res_time.append(end - start)
        
        ###################################################################################################################################OMPSC algorithm results
        start = time.process_time()
        Row_labels_OMPSC,Col_labels_OMPSC=Competitive_algorithms.OMPSC_algorithm(A,K_1,K_2,n,m)
        end = time.process_time()
        Res_row.append(normalized_mutual_info_score(ground_truth_cluster_label_1,Row_labels_OMPSC))
        Res_col.append(normalized_mutual_info_score(ground_truth_cluster_label_2,Col_labels_OMPSC))
        Res_time.append(end - start)
        
        ###################################################################################################################################COSSC algorithm results
        start = time.process_time()
        Row_labels_COSSC,Col_labels_COSSC=Competitive_algorithms.COSSC_algorithm(A,K_1,K_2)
        end = time.process_time()
        Res_row.append(normalized_mutual_info_score(ground_truth_cluster_label_1,Row_labels_COSSC))
        Res_col.append(normalized_mutual_info_score(ground_truth_cluster_label_2,Col_labels_COSSC))
        Res_time.append(end - start)

        ###################################################################################################################################INSC algorithm results
        start = time.process_time()
        Row_labels_INSC,Col_labels_INSC=Competitive_algorithms.INSC_algorithm(A,K_1,K_2,n,m)
        end = time.process_time()
        Res_row.append(normalized_mutual_info_score(ground_truth_cluster_label_1,Row_labels_INSC))
        Res_col.append(normalized_mutual_info_score(ground_truth_cluster_label_2,Col_labels_INSC))
        Res_time.append(end - start)

        ###################################################################################################################################SVDK algorithm results
        start = time.process_time()
        Row_labels_SVDK,Col_labels_SVDK=Competitive_algorithms.SVDK_algorithm(A,K_1,K_2)
        end = time.process_time()
        Res_row.append(normalized_mutual_info_score(ground_truth_cluster_label_1,Row_labels_SVDK))
        Res_col.append(normalized_mutual_info_score(ground_truth_cluster_label_2,Col_labels_SVDK))
        Res_time.append(end - start)
        NMI_res_row.append(Res_row)
        NMI_res_col.append(Res_col)
        Time_res.append(Res_time)
        
       

