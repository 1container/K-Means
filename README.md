公式无法显示，请忽略或下载查看。
# K-Means
​        运用K-means算法在西瓜数据集上进行聚类，选定k=2，将西瓜分为好瓜/坏瓜两个聚类簇，进行可视化效果展示和代码呈现。并尝试对初始化中心的选择进行优化。

###  一、使用K-Means算法对西瓜数据集聚类

####  可视化效果

![图1 可视化](https://github.com/1container/K-Means/blob/main/images/%E5%9B%BE1%20%E5%8F%AF%E8%A7%86%E5%8C%96.png)

​        可以看出，算法对西瓜数据集分为两类。

###  二、不同随机初始化中心对结果的影响

####  分类性能评价指标定义

​        不同的随机初始化中心对分类性能有影响。对分类性能的**内部评价**通常采用**类间分离度**和**类内紧密度**两个指标。基于簇与簇之间距离尽可能远，簇中元素尽可能接近的原则，利用**类间距离**和**类内距离**对分类性能做出评价。

​        定义类间距离为：
$$distbet=\frac{1}{K}\left(\sum_{k=1}^K\parallel{c_k}-{\bar{x}}\parallel_2\right)$$
​        其中 $K$ 是样本的总簇数，$\parallel{c_k}$ 是簇 $k$ 的均值向量, ${\bar{x}}$ 是所有样本的均值向量。

​        定义类内距离为：
$$distin=\sum_{k=1}^K\frac{1}{n}\left(\sum_{i=1}^{n}\parallel {x_i}-{c_k}\parallel_2\right)$$
​        其中 $n$ 是簇 $k$ 的样本总数，${x_i}$ 是簇 $k$ 的第 $i$ 个向量。

​        以类间距离、类内距离二者之商为指标度量分类性能，度量指标为：
$$I=\frac{distbet}{distin}$$
​        类间距越大越好，类内距越小越好，由定义可知，该指标越大，聚类效果越好。

####  不同的随机初始化中心对结果的影响

​        现考察不同的随机初始化中心对分类性能的影响。西瓜数据集在去除重复数据后共30个数据，对所有可能的随机初始化中心进行穷举，共有 $$29+28+27+...+1=435$$ 种随机初始化中心组合。以这435种组合为初始化中心依次进行聚类，聚类效果通过上述度量指标 $$I$$ 进行评估。所有组合的度量指标散点图如下所示。

![图2 指数散点图](https://github.com/1container/K-Means/blob/main/images/%E5%9B%BE2%20%E6%8C%87%E6%95%B0%E6%95%A3%E7%82%B9%E5%9B%BE.png)

​        从图中可以看出，不同的初始化中心会产生不同的分类效果，随机选择初始化中心，有一定概率得出效果较差的聚类结果。

​        该部分代码实现如下。

```python
# 计算簇间距离
def distbet(matmean, data):
    allmean=np.mean(data,axis=0)
    mean=np.tile(allmean,(2,1))
    return np.sum(np.sqrt(np.sum(np.square(matmean-mean),axis=1)))/2#行相加开根号再相加
# 计算簇内距离
def distin(matmean, cl):#相关变量见上一部分代码
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
```

###  三、初始化中心选择方法优化

​        不同的初始聚类中心对分类性能有不同影响，一般来说，较好的初始化中心应位于每一簇中心附近，这样可降低算法迭代次数并避免陷入局部最优。

​        **传统K-means算法**：随机选择初始化中心。生成初始化中心快，迭代次数较多。如果随机初始化中心处于同一簇内，会导致较差的聚类结果。

​        **K-means++算法**：随机选取第一个初始化中心，距第一个簇中心越远的样本点有越大可能被选为下一个簇中心。

​        代码实现如下。

```python
#初始化中心选择
def fsmean(data,dist=distEclud):
    firmean=data[np.random.randint(le)]# 随机选取一个初始化中心
    print(firmean)
    maxdist=0;index=-1
    for i in range(le):
    # 计算样本与均值向量的距离，选择距离最大的样本为第二个簇中心
        disti = dist(firmean,data[i])
        if disti > maxdist:
            print(i)
            maxdist = disti
            index=i
    return np.vstack((firmean,data[index]))
```

​        在实现对初始化中心的改进后，穷举所有的初始化向量组合，其性能评估如下。

![图3 K-means++](https://github.com/1container/K-Means/blob/main/images/%E5%9B%BE3%20K-means%2B%2B.png)

​        从K-means++算法的性能评估图来看，仅基于初始化中心选择方式的改进，度量指标下限明显提高，聚类效果有显著提升。

**参考资料**

[1]谢娟英,周颖,王明钊,姜炜亮.聚类有效性评价新指标[J].智能系统学报,2017,12(06):873-882.

[2] [Kmeans++聚类算法原理与实现 - 知乎 (zhihu.com)

[3] 周志华.机器学习[M].北京:清华大学出版社,2016.
