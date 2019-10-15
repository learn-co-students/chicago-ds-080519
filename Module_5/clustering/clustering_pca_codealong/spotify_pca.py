import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def preprocess_song_data(df):

    features = df.copy()

    # Will need artists/songs for plotting, but not for clustering
    features.drop(['id', 'name', 'artists'], axis = 1, inplace = True)
    column_names = features.columns

    # All features are numerical; standard scaling allows for effective clustering.
    ss = StandardScaler()
    features = pd.DataFrame(ss.fit_transform(features))
    features.columns = column_names
    
    return features



def km_fit_plot(clusterer, features, artist_names, n_pca_components, show_2d_names = False):
    
    
    clusterer.fit(features)
    
    pca = PCA(n_components = n_pca_components)
    
    pca_features = pd.DataFrame(pca.fit_transform(features))
    pca_features['labels'] = clusterer.labels_
    pca_features['artist'] = artist_names
    
    label_list = list(np.unique(clusterer.labels_))


    
    if n_pca_components == 2 and show_2d_names == False:
        fig = plt.figure(figsize = (20,20))
        ax = fig.add_subplot(111)
        for label in label_list:
            label_df = pca_features[pca_features.labels == label]
            ax.scatter(label_df[0], label_df[1], label = label)
        for i, name in enumerate(pca_features['artist']):
            ax.annotate(name, (pca_features[0][i], pca_features[1][i]))
            
        plt.legend()
        plt.show()
        
    elif n_pca_components == 2 and show_2d_names == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label in label_list:
            label_df = pca_features[pca_features.labels == label]
            ax.scatter(label_df[0], label_df[1], label = label)
        
        plt.legend()
        plt.show()
    
    elif n_pca_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for label in label_list:
            label_df = pca_features[pca_features.labels == label]
            ax.scatter(label_df[0], label_df[1], label_df[2], label = label)
        
        plt.legend()
        plt.show()
        
    else:
        print("too few or too many components for plotting")
    

    

