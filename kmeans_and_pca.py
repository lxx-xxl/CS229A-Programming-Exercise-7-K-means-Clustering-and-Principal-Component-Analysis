# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:37:28 2022

@author: taylo
"""

import os
import scipy.io as scio
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from numpy import random

#example 1 (看c对decision boundary的影响)用linear kernel
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\7-kmeans_and_pca\data')
data = scio.loadmat('ex7data2.mat') #读取出来的data是字典格式
X=data['X']
x1=X[:,0]
x2=X[:,1]

#k-means
#kmeans=KMeans(n_clusters=3)
#kmeans.fit(X)
#print (kmeans.labels_)

#plot
#plt.figure(figsize=(8,10))
#colors=['r', 'b','g']
#markers=['o','s','D']
#for i,l in enumerate(kmeans.labels_):   
#     plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')  
#plt.show()  

def cluster_assignment(X,centroid):
    idx=[]  #放c(i)
    for i in range(len(X)): #0到299
        distance=np.linalg.norm((X[i]-centroid),axis=1)
        idx_i=np.argmin(distance)   #0 1 2 
        idx.append(idx_i)
    return np.array(idx)    #返回每个样本所属的类别
init_centroid=np.array([[3,3],[6,2],[8,5]])
idx=cluster_assignment(X, init_centroid)    #每个样本所属的类别

#plt.scatter(x1,x2,c=idx,cmap='rainbow')
#plt.scatter(init_centroid[:,0], init_centroid[:,1],s=200,marker='x',c=[0,1,2],cmap='rainbow')

def compute_centroids(X,idx,k):
    centroids=[]    #存放centroid means
    for i in range(k):
        centroids_i=np.mean(X[idx==i],axis=0)
        centroids.append(centroids_i)
    return np.array(centroids)
centroids=compute_centroids(X,idx,3)    #三个聚类中心由初始值变成当前平均值

#plt.scatter(centroids[:,0], centroids[:,1],s=200,marker='+',c='k')

def cost_function(X,centroids,idx,k):
    costs=[]
    for i in range(k):
        c=[]
        for x in range(X[idx==i].shape[0]):
            c.append(centroids[i])
        inner=X[idx==i]-c
        cost=np.sum(np.power(inner,2))
        costs.append(cost)
    return np.sum(np.array(costs))/(X.shape[0])

def run_kmeans(X,iter,init_c):
    k=len(init_c)
    centroids=[]
    centroid=init_c
    centroids.append(centroid)
    costs=[]
    for i in range(iter):
        idx=cluster_assignment(X,centroid)
        centroid=compute_centroids(X,idx,k)
        cost=cost_function(X,centroid,idx,k)
        centroids.append(centroid)  #得到三维矩阵： iter classification 每个class的mean
        costs.append(cost)
    return np.array(costs),np.array(centroids),idx
        
def plot_2d(X,idx,centroids):
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=idx,cmap='rainbow')
    plt.plot(centroids[:,:,0], centroids[:,:,1],'kx--')
    plt.scatter(centroids[10,:,0], centroids[10,:,1],s=100,c='k',marker='o')
    plt.show()
costs,centroids,idx=run_kmeans(X,iter=10,init_c=init_centroid)
#plot_2d(X,idx,centroids)
print(costs)

#random initialization
def init_centroids(X,K):
    index=np.random.choice(len(X),K)    #左闭右开
    return X[index]

#for i in range(4):
#    costs,centroids,idx=run_kmeans(X,iter=10,init_c=init_centroids(X,3))
#    plot_2d(X,idx,centroids)

image=io.imread(r'bird_small.png')
#plt.imshow(image)
data_bird = scio.loadmat('bird_small.mat')
A=data_bird['A']

#mean normalization
A=A/255 #三原色各有255个色度可选
A=A.reshape(-1,3)   #3列，-1是指要求3列情况下，自动计算行数,从128x128x3变成128*128x3
#因为128x128是一张图形式的像素点排列就是把128*128个像素点例子按128x128的央视排列，可以看看上面那个图横纵左边都是128
#128X128X3指的就是128作为x和y轴来定位图片上的128*128个例子，3是RGB三个通道各自的强度
#所以放在kmeans里面时要把全部例子展开
#costs_bird,centroids_bird,idx_bird=run_kmeans(A, 20, init_c=init_centroids(A,16))
#centroids_b=centroids_bird[-1]  #取最后一次的centroid mean
#k=16
#im=np.zeros(A.shape)
#for i in range(k):
#    im[idx_bird==i]=centroids_b[i]
#im=im.reshape(128,128,3)    #区别就是用kmean之前RGN每个通道有256个选择 用了之后就只有16个选择了
#plt.imshow(im)

#Notice that you have significantly reduced the
#number of bits that are required to describe the image
#The original image
#required 24 bits for each one of the 128×128 pixel locations, resulting in total
#size of 128 × 128 × 24 = 393, 216 bits. 
#The new representation requires some
#overhead storage in form of a dictionary of 16 colors, each of which require
#24 bits, but the image itself then only requires 4 bits per pixel location. The
#final number of bits used is therefore 16 × 24 + 128 × 128 × 4 = 65, 920 bits


#PCA

data_PCA = scio.loadmat('ex7data1.mat')
X_PCA=data_PCA['X'] #50,2
#plt.figure()    #要是想跟下面的分开俩图显示，那就用这个
#plt.scatter(X_PCA[:,0],X_PCA[:,1])

#normalization，使得example均值为0
mean_pca=np.mean(X_PCA,axis=0)
X_pca=X_PCA-mean_pca
#plt.figure()
#plt.scatter(X_pca[:,0],X_pca[:,1])

#get sigma
def compute_covariancematrix(X):
    sigma=np.dot(X.T,X)/len(X)
    return sigma
#get u s v
sigma=compute_covariancematrix(X_pca)
u,s,v=np.linalg.svd(sigma)  #u是n*n
#plt.figure(figsize=(7,7))
#plt.scatter(X_pca[:,0],X_pca[:,1])
#plt.plot([u[0,0],0],[u[1,0],0],c='r')
#plt.plot([u[0,1],0],[u[1,1],0],c='k')

#get z
u_reduce=u[:,0] #取第一列(2,)
u_reduce=u_reduce.reshape(2,1)  #(2,1)
z=np.dot(X_pca,u_reduce)  #(50,2)*(2,1)=(50,1)
print(z.shape)

# approximately recover the data by projecting them back onto the original high
#dimensional space
X_approx=np.dot(z,u_reduce.T)   #(50,1)*(1,2)=(50,2)
#想还原到没mean normalization之前的就把mean加回去就行（X_pca和X_approx）

#visualize origin in blue, approx in red
#plt.figure(figsize=(7,7))
#plt.scatter(X_pca[:,0],X_pca[:,1],c='b',marker='o')
#plt.scatter(X_approx[:,0],X_approx[:,1],c='r',marker='o')
#for i in range(len(X_pca)):
#    plt.plot([X_pca[i,0],X_approx[i,0]],[X_pca[i,1],X_approx[i,1]],'--',c='k')
    
#face image

data_face = scio.loadmat('ex7faces.mat')
face=data_face['X'] #5000,1024(32*32做成人脸，5000个例子)
#visualize face
def plot_100face(X):
    fig,axs=plt.subplots(ncols=10,nrows=10,figsize=(10,10)) #10行和10列框框，一共100个框框10*10排列，figsize是每个框框的大小
    for c in range(10):
        for r in range(10):
            axs[c,r].imshow(X[10*c+r].reshape(32,32).T,cmap='Greys_r')
            axs[c,r].set_xticks([])
            axs[c,r].set_yticks([])

#plot_100face(face)

#normalization
mean_face=np.mean(face,axis=0)
X_face=face-mean_face
sigma_face=compute_covariancematrix(X_face)
u_f,s_f,v_f=np.linalg.svd(sigma_face)
#plot_100face(u_f)
#displays the first 36 principal components that describe the largest variations
u_reduce_f=u_f[:,:36]
z_f=np.dot(X_face,u_reduce_f)   #5000,1024变成5000,36
X_approx_f=np.dot(z_f,u_reduce_f.T)
plot_100face(X_approx_f)