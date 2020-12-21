# import liberares
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#------------------------------------

# select random points
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    random_centroids = np.random.randint(0, m, k)  

    for i in range(k): # 0 , 1 , 2
        centroids[i,:] = X[random_centroids[i],:] 

    return centroids


# centroid function
def find_closest_centroids(X, centroids):
     m = X.shape[0] # 300
     k = centroids.shape[0] # 3
     centroids_list= np.zeros(m) # ليست من 300 صقر وكل صفر هيتغير حسب هو تبع انهي سنتر


     for i in range(m): # 0 ,1 ,2 ,.....,299
         min_dist = 10000
         for j in range(k): # 0 ,2 ,3 #عدد السنترز # (j) هي رقم السنتر 
             dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
             if dist < min_dist:
                 min_dist = dist
                 centroids_list[i] = j
                 
     return centroids_list

# centroid maker
def displacement_centroids(X, centroids_list, k):
    m, n = X.shape  #300 , 2
    centroids = np.zeros((k, n))  #3x2

    for i in range(k): #0, 1, 2  # 
        indices = np.where(centroids_list == i) 
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
# هاخد السنتر الاول الي هو 0 واجيب كل النقط اللي معاه واقسمها علي عدد النقط وهي دي هتبقي قيمه السنتر الاول الجديده وهكذا
    return centroids

# k means function
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = displacement_centroids(X, idx, k)
        
    return idx, centroids
#------------------------------------

# load data
data = loadmat('D://Programing//ML//Unsupervised ML//ex7data2.mat')
print(data)
print(data['X'])
print(data['X'].shape)
X = data['X']

#ploting data
fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(X[ : , 0], X[ : ,1], s=30, color='blue', label='Cluster 1')

# Selection part   #خطوه تخصيص السنترز
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]]) #تخصيص السنترز يدويا
#initial_centroids = np.array([[8, 0], [8, 6], [0, 3]]) #تخصيص السنترز يدويا
#initial_centroids = init_centroids(X, 3) #تخصيص السنترز عشوائا
# التحديد بالنظر افضل من التحديد العشوائي
print(initial_centroids )

# Select the centroids
FCC = find_closest_centroids(X, initial_centroids)
print(FCC)

# displacement part  # خطوه ازاحه السنترز
c = displacement_centroids(X, FCC, 3)
print(c)

# Repeating selection and displacement
# ممكن نعوض عن الخطوتين اللي فاتوا بالخطوخ دي بس 
for x in range(10):
    #apply k means
    idx, centroids = run_k_means(X, initial_centroids, x)
    print(idx)
    print()
    print(centroids )
    
    # draw it
    cluster1 = X[np.where(idx == 0)[0],:]
    cluster2 = X[np.where(idx == 1)[0],:]
    cluster3 = X[np.where(idx == 2)[0],:]
    
    fig, ax = plt.subplots(figsize=(9,6))
    ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
    ax.scatter(centroids[0,0],centroids[0,1],s=300, color='r')
    
    ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
    ax.scatter(centroids[1,0],centroids[1,1],s=300, color='g')
    
    ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
    ax.scatter(centroids[2,0],centroids[2,1],s=300, color='b')
    
    ax.legend()
