import math
import pandas as pd 
import numpy as np 
# 读取数据
import csv
# with open('F:\\watermelon.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]

# # 数据处理
# file = np.array(rows); file1 = file[1:,1:]#去除第一列第一行
# file2=pd.DataFrame(file1)# 去除重复数据
# file2.drop_duplicates(inplace=True)
# file3=np.array(file2); data = file3.astype(float)# 转换格式

file=pd.read_csv('F:\\watermelon.csv')#读取数据
file1=file.drop('number',axis=1)#删除number列
file1.drop_duplicates(inplace=True)#去除重复数据
data=np.array(file)#转换格式

# 计算两向量的欧氏距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.square(vecA-vecB)))

# 循环
def kMeans(data,matmean,dist=distEclud):
    le=len(data)
    cl = np.mat(np.zeros((le,2)))#创建结果分类矩阵
    for q in range(999):
        for i in range(le):
            mindist=99999;index=-1
        # 计算样本与均值向量的距离，确定样本簇标记
            for j in range(2):
                distji = dist(matmean[j],data[i])
                if distji < mindist:
                    mindist = distji;index = j
            cl[i,:]=index,mindist**2
    # 计算新均值向量
        mattest = np.mat(np.zeros((2,2)))
        for p in range(2):
            clust = data[np.where(cl[:,0]==p)[0]]
            mattest[p,:] = np.mean(clust,axis=0)#若将mattest初始化放于q的for循环外，此处第二次循环matmean随mattest变化
        if (mattest == matmean).all() == True:#均值向量未改变
            break
        else:
            matmean=mattest
        #迭代第q次结果与q-1次相同
    sumcl=np.sum(cl,axis=0)
    sse=sumcl[0,1]
    return [matmean,cl,q-1]

# 随机选取2个样本作为初始均向量
import random
le=np.shape(data)[0]
s = random.sample(range(le),2)
randvec=np.vstack((data[s[0]],data[s[1]]))

a=kMeans(data,randvec)
matmean=a[0]
cl=a[1]

#可视化聚类结果
import matplotlib.pyplot as plt
for i in range(le):
    if cl[i,0]==0:
        plt.plot(data[i,0],data[i,1],'sg')
    elif cl[i,0]==1:
        plt.plot(data[i,0],data[i,1],'or')
plt.plot(matmean[0,0],matmean[0,1],'.k')
plt.plot(matmean[1,0],matmean[1,1],'.k')
plt.title('Classification of watermelons')
plt.xlabel('density')
plt.ylabel('sugercontent')
plt.grid(c='silver',linestyle='--')
plt.xlim([0.2,0.9])  # x轴边界
plt.ylim([0,0.8])  # y轴边界
plt.show()
