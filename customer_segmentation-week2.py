import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#loading the data
data = pd.read_excel("Online Retail.xlsx")
print(data.head())

#cleaning
data=data.dropna()
print(data)
data=data.drop_duplicates()
data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
data['Total_Spend']=data['Quantity']*data['UnitPrice']
summary=data.groupby('CustomerID').agg({
    'Total_Spend':'sum',
    'InvoiceDate':'max',
    'CustomerID':'count'
    }).rename(columns={'CustomerID':'Frequency'})
summary['Recency']=(pd.Timestamp.now()-summary['InvoiceDate']).dt.days
print(summary.head())

#cluster algorithms
feature=summary[['Total_Spend','Recency','Frequency']]
scaler=StandardScaler()
scaled_features=scaler.fit_transform(feature)

 #k-Means
inertia=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11),inertia,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")     
plt.show()
optimal_cluster=4
kmeans=KMeans(n_clusters=optimal_cluster,random_state=42)
summary['cluster']=kmeans.fit_predict(scaled_features)


#visual cluster
pca=PCA(n_components=2)
reduced_feature=pca.fit_transform(scaled_features)

plt.scatter(reduced_feature[:,0],reduced_feature[:,1],c=summary['cluster'],cmap="viridis")
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.title('customer segments')
plt.colorbar(label="cluster")
plt.show()

#marketing strategies

cluster_characteristics=summary.groupby('cluster').mean()
print(cluster_characteristics)