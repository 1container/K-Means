import math
import pandas as pd 
import numpy as np 
# 读取数据
import csv
with open('F:\\watermelon.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
# 数据处理
file = np.array(rows); file1 = file[1:,1:]#去除第一列第一行
file2=pd.DataFrame(file1)# 去除重复数据
file2.drop_duplicates(inplace=True)
file3=np.array(file2); data = file3.astype(float)# 转换格式
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
            cl[i,:]=index,mindist
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
    return [matmean,cl,sse,q-1]

# 计算簇间距离
def distbet(matmean, data):
    allmean=np.mean(data,axis=0)
    mean=np.tile(allmean,(2,1))
    return np.sum(np.sqrt(np.sum(np.square(matmean-mean),axis=1)))/2#行相加开根号再相加
# 计算簇内距离
def distin(matmean, cl):
    num0=np.sum(cl[:,0]==0)
    num1=np.sum(cl[:,0]==1)
    dist0=np.sum(cl[np.where(cl[:,0]==0)[0],1])
    dist1=np.sum(cl[np.where(cl[:,0]==1)[0],1])
    return dist0/num0 + dist1/num1

# 穷举随机组合 得到相应度量指标
ass=[]
for i in range(30):
    for j in range(i+1,30):
        randvec=np.mat(np.zeros((2,2)))
        randvec=np.vstack((data[i],data[j]))
        a=kMeans(data,randvec)
        matmean=a[0]
        cl=a[1]
        x=distbet(matmean,data)
        y=distin(matmean,cl)
        disass=distbet(matmean,data)/distin(matmean,cl)
        ass.append(disass)#评价指标形成列表
print(max(ass),min(ass))

# 指标散点图
x=range(len(ass))
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#图片显示中文，参考网址：https://blog.csdn.net/seveneleve/article/details/108974201
p1=plt.scatter(x,ass,marker='.')
plt.title('西瓜数据集穷举初始化中心的分类性能指数散点图')
plt.xlabel("编号")
plt.ylabel("指数")
plt.show()
