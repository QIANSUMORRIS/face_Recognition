#coding:utf-8

from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people#下载数据集
from sklearn.model_selection import GridSearchCV#调参
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import sys
# from PyQt import QtGui 

# #展现程序运行记录 INFO
logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(message)s')
logger = logging.getLogger()

logger.info("开始打印日志")
logger.debug("开始人脸数据提取")
#====================================================
#开始下载人脸数据
#====================================================
#每个人的图片至少有70涨图片，像素改为原来的0.4倍，resize取值范围为0~1
lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)#降维

#print(lfw_people.keys())
#===============================
#提取数据集的信息以及x特征
#===============================
n_samples, h, w = lfw_people.images.shape #(1288, 50, 37)
X = lfw_people.data #(1288, 1850)
n_features = X.shape[1]#特征值
#=============================
#提取数据集的y标签
#=============================
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]#1288个人中只有7个名字
#=============================
#打印数据集的信息
#============================= 
print("数据集信息如下:")
print("总实例个数为: %d" % n_samples) #1288
print("每个实例特征维度是: %d" % n_features) #1850
print("总共有  %d 类" % n_classes) #7
#=================================
#获取X， 和y之后，对数据集进行切分
#=================================   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.25)
#=================================
#先对数据集进行降维，之后再带入模型训练
#=================================
n_components = 150#pca降维至150
print("把 %d个实例降维至 %d"% ((X_train.shape[0]), n_components))
t0 = time()
pca = PCA(svd_solver='randomized', n_components = n_components, whiten = True).fit(X_train)#Randomized
print("PCA降维使用的时间是 %0.3fs" % (time() - t0))
          
eigenfaces = pca.components_.reshape((n_components, h, w))         
print("开始对训练集进行降维特征提取")
t0 = time()
X_train_pca = pca.transform(X_train)#降唯处理
X_test_pca = pca.transform(X_test)#降唯处理
print("共耗时 %0.3fs" % (time() - t0))
#============================    
#开始调参及模型训练
#============================     
print("开始利用训练集调参并训练模型")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], } #特征点使用的比例
clf = GridSearchCV(SVC(kernel = 'rbf', class_weight='balanced'), param_grid) #balanced
clf = clf.fit (X_train_pca, y_train)
print("训练模型共耗时%0.3fs" % (time() - t0))
print("最好的模型是:")
print(clf.best_estimator_)
#=========================
#开始预测
#========================= 
print("预测测试集中人物图片的名称 ")
t0=time()
y_pred = clf.predict(X_test_pca)
print ("预测共耗时 %0.3fs" % (time() - t0))
#=========================
#评估模型
#=========================          
print(classification_report(y_test, y_pred, target_names = target_names))
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))  
#=============================  
#绘制图像集中每个图像的正确vs预测标签
#=============================                                          
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return r'pred:%s\true:%s' % (pred_name, true_name)
#============================
#实际上只循环前12个
#============================       
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
#======================================
#构造特征脸和数据集中真实脸的对比图
#======================================       
def plot_gallery(images, titles, h, w, n_row = 3, n_col = 4):
    """ 构造绘制特征连和真实连肖像的函数"""   
    plt.figure(figsize = (1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom = 0, left = .01, right = .99, top = .90, hspace = .35) 
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray)
        plt.title(titles[i], size = 12)
        plt.xticks(())
        plt.yticks(())
          
plot_gallery(X_test, prediction_titles, h, w)
          
#plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface%d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
          
plt.show()
