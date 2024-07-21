from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
X=[
    [ 0.697, 0.46 ],
    [ 0.774, 0.376],
    [ 0.634, 0.264],
    [ 0.608, 0.318],
    [ 0.556, 0.215],
    [ 0.403, 0.237],
    [ 0.481, 0.149],
    [ 0.437, 0.211],
    [ 0.666, 0.091],
    [ 0.243, 0.267],
    [ 0.245, 0.057],
    [ 0.343, 0.099],
    [ 0.639, 0.161],
    [ 0.657, 0.198],
    [ 0.36 , 0.37 ],
    [ 0.593, 0.042],
    [ 0.719, 0.103]
]
y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
c=5
str_c=str(c)
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.title("Watermelon dataset v3.0")
for i in range(0, len(X)):
    if y[i] == 1:
        plt.scatter(X[i][0],X[i][1],c='r',marker='*')
    else:
        plt.scatter(X[i][0],X[i][1],c='g',marker='*')


print("-"*20+"线性核"+"-"*20)
clf1=svm.SVC(C = c,kernel='linear')
print("交叉验证评分",cross_val_score(clf1,X,y,cv=5,scoring='accuracy').mean())
clf1.fit(X,y)
print("支持向量数目:",clf1.n_support_.sum())
print("支持向量:")
print(clf1.support_vectors_)

plt.subplot(1,3,2)
plt.title("Linear kernel(C="+str_c+")")
for i in X:
    res=clf1.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='*')
    else :
        plt.scatter(i[0],i[1],c='g',marker='*')

print("-"*20+"高斯核"+"-"*20)
clf2=svm.SVC(C=c,kernel='rbf')
print("交叉验证评分",cross_val_score(clf2,X,y,cv=5,scoring='accuracy').mean())
clf2.fit(X,y)
print("支持向量数目",clf2.n_support_.sum())
print("支持向量:")
print(clf2.support_vectors_)


plt.subplot(1,3,3)
plt.title("Gaussian kernel(C="+str_c+")")
for i in X:
    res=clf2.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='*')
    else :
        plt.scatter(i[0],i[1],c='g',marker='*')
plt.show()