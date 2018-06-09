#coding:utf-8
'''
@author: chencheng, topzero.cn
@date  : 20180609
@brief :
    compare the gmm and kmeans
@refer :
    https://www.jianshu.com/p/a4d8fa39c762
'''


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


''' get features '''
def getFeatureFromData(datafile):
    data = pd.read_excel(datafile)
    # print("各字段缺失情况: \n", data.isnull().sum())
    if (0):
        fig = plt.figure(figsize=(16, 9))
        for i, col in enumerate(list(data.columns)[1:]):
            plt.subplot(321+i)
            q95 = np.percentile(data[col], 95)
            sns.distplot(data[data[col] < q95][col])
        plt.show()

    features = data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']]
    # 剔除极值或异常值
    ids = []
    for i in list(features.columns):
        q1 = np.percentile(features[i], 25)
        q3 = np.percentile(features[i], 75)
        intervel = 1.6*(q3 - q1)/2
        low = q1 - intervel
        high = q3 + intervel
        ids.extend(list(features[(features[i] <= low) |
                                 (features[i] >= high)].index))
    ids = list(set(ids))
    features = features.drop(ids)

    return features


''' PCA decrease dimension'''
def PCA(features):
    # 计算每一列的平均值
    meandata = np.mean(features, axis=0)  
    # 均值归一化
    features = features - meandata    
    # 求协方差矩阵
    cov = np.cov(features.transpose())
    # 求解特征值和特征向量
    eigVals, eigVectors = np.linalg.eig(cov) 
    # 选择前两个特征向量
    pca_mat = eigVectors[:, :2]
    pca_data = np.dot(features , pca_mat)
    pca_data = pd.DataFrame(pca_data, columns=['pca1', 'pca2'])

    # 两个主成分的散点图
    if (0):
        plt.subplot(111)
        plt.scatter(pca_data['pca1'], pca_data['pca2'])
        plt.xlabel('pca_1')
        plt.ylabel('pca_2')
        plt.show()

    return pca_data


def clustering(pca_data):
    score_kmean = []
    score_gmm = []
    random_state = 87
    n_cluster = np.arange(2, 6)

    for i, k in zip([0, 2, 4, 6], n_cluster):
        # K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        
        cluster1 = kmeans.fit_predict(pca_data)
        score_kmean.append(silhouette_score(pca_data, cluster1))

        # gmm聚类
        # covariance_type can be 'spherical, tied, diag, full'
        gmm = GaussianMixture(n_components=k, covariance_type='spherical', random_state=random_state)

        cluster2 = gmm.fit(pca_data).predict(pca_data)
        score_gmm.append(silhouette_score(pca_data, cluster2))

        # 聚类效果图
        plt.subplot(421+i)
        plt.scatter(pca_data['pca1'], pca_data['pca2'], c=cluster1, cmap=plt.cm.Paired)
        if i == 6:
            plt.xlabel('K-means')

        plt.subplot(421+i+1)
        plt.scatter(pca_data['pca1'], pca_data['pca2'], c=cluster2, cmap=plt.cm.Paired)
        if i == 6:
            plt.xlabel('GMM')

    plt.show()

    # 得分变化对比图
    sil_score = pd.DataFrame({'k': np.arange(2, 6),
                              'score_kmean': score_kmean,
                              'score_gmm': score_gmm})
    # K-means和GMM得分对比
    plt.figure(figsize=(10, 6))
    plt.bar(sil_score['k']-0.15, sil_score['score_kmean'], width=0.3,
            facecolor='blue', label='Kmeans_score')
    plt.bar(sil_score['k']+0.15, sil_score['score_gmm'], width=0.3,
            facecolor='green', label='GMM_score')
    plt.xticks(np.arange(2, 6))
    plt.legend(fontsize=16)
    plt.ylabel('silhouette_score', fontsize=16)
    plt.xlabel('k')
    plt.show()



def main():
    datafile = '../../data/clustering_data/customer_data.xlsx'
    features = getFeatureFromData(datafile)
    # print(type(features))

    pca_data = PCA(features)
    # print(type(pca_data))

    clustering(pca_data)


if __name__ == "__main__":
    main()



