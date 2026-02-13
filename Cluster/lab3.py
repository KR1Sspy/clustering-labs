import numpy as np
from sklearn.cluster import KMeans
def kmeans(data:np.ndarray,number_classes:int):
    r,c = data.shape
    label = np.zeros([1,c],dtype=np.int8)
    centroid = np.zeros([r,number_classes])
    for i in range(number_classes):
        label[0,i] = i+1
        centroid[:,i] = data[:,i] 
    while True:
        old_label = label.copy()
        centroid_accum = np.zeros([r,number_classes])
        group_num = np.zeros([1,number_classes])
        for i in range(c):
            mid = np.zeros([1,number_classes])
            for j in range(number_classes):
                mid[0,j] = np.linalg.norm(data[:,i]-centroid[:,j])
            label[0,i] = np.argmin(mid)+1
            centroid_accum[:,np.argmin(mid)] = centroid_accum[:,np.argmin(mid)] + data[:,i]
            group_num[0,np.argmin(mid)] = group_num[0,np.argmin(mid)]+1
        centroid = centroid_accum/group_num
        if (label == old_label).all():
            break
    return label,centroid
if __name__ =="__main__":
    data = np.array([[0,0,7,5,5],[2,0,3,0,1]])
    k = 2
    label,centroid = kmeans(data,k)
    print(label)
    print(centroid)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data.T)
    print(kmeans.labels_+1)
    print(kmeans.cluster_centers_.T)
    
   
    
        
        
            
                
                
    
    