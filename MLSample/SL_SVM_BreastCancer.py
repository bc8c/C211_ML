import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

cancer = load_breast_cancer()

malignant = cancer.data[cancer.target==0] # 음성 37%
benign = cancer.data[cancer.target==1]    # 양성 63%

print(malignant.shape, benign.shape)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

model = SVC()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(train_score, test_score)

pred_y = model.predict(X_test)
print(pred_y)
print(y_test)

fig=plt.figure(figsize=[10,8])
plt.title('Cancer - boxplot for features',fontsize=15)
plt.boxplot(cancer.data)
plt.xticks(np.arange(30)+1,cancer.feature_names,rotation=90)
plt.xlabel('features')
plt.ylabel('scale')
plt.show()

# 균등 비율
X_max = X_train.max(axis=0)
X_min = X_train.min(axis=0)
X_train_uni = (X_train - X_min) / (X_max - X_min)
X_test_uni = (X_test - X_min) / (X_max - X_min) # 학습데이터 값을 기준으로 변환

plt.figure(figsize=[8,8])
plt.boxplot(X_train_uni)
plt.xticks(np.arange(30)+1,cancer.feature_names,rotation=90)
plt.show()

# 정규분포
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std # 학습데이터 값을 기준으로 변환

plt.figure(figsize=[8,8])
plt.boxplot(X_train_norm)
plt.xticks(np.arange(30)+1,cancer.feature_names,rotation=90)
plt.show()

# 균등 정규화 학습
model = SVC()
model.fit(X_train_uni, y_train)

train_score = model.score(X_train_uni, y_train)
test_score = model.score(X_test_uni, y_test)
print(train_score, test_score)

# 정규분포 정규화 학습
model = SVC()
model.fit(X_train_norm, y_train)

train_score = model.score(X_train_norm, y_train)
test_score = model.score(X_test_norm, y_test)
print(train_score, test_score)

pred_y=model.predict(X_test_norm)
print(np.where(y_test!=pred_y))


# 하이퍼파라미터C와 gamma 값을 변화시키며 최적의 값 찾아내기

C=[0.1,1,10,100]
gamma=[0.001,0.01,0.05,0.1,0.3]
s_train=[]
s_test=[]

for c in C:
    s1=[]
    s2=[]
    
    for g in gamma:
        model=SVC(C=c,gamma=g)
        model.fit(X_train_norm,y_train)

        pred_y=model.predict(X_test_norm)
        s1.append(model.score(X_train_norm,y_train))
        s2.append(model.score(X_test_norm,y_test))
        
    s_train.append(s1)
    s_test.append(s2)
    
fig=plt.figure(figsize=[12,8])
for i in range(len(C)):
    plt.subplot(1,len(C),i+1)
    plt.plot(s_train[i],'gs--',label='train')
    plt.plot(s_test[i],'ro-',label='test')
    plt.title('C= %f' % (C[i]))
    plt.xticks(range(len(gamma)),gamma)
    plt.ylim(0,1)
    plt.xlabel('gamma')
    plt.ylabel('score')
    plt.legend(loc='lower right')
plt.show()


# 2차원 시각화를 위해 특성 2개만 사용하여 학습해봄
col1 = 0 #20
col2 = 1 #27
# 컬럼은 하나하나 상관관계 등 여러가지를 고려하여 선택을 해야한다.


X = cancer.data[:,[col1,col2]]
y = cancer.target

X_train,X_test,y_train,y_test = train_test_split(X, y)

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train_norm = (X_train-X_mean)/X_std
X_test_norm = (X_test-X_mean)/X_std

model = SVC(C=1, gamma=0.05, probability=True) # 확률 점수를 얻기 위해 probability=True
model.fit(X_train_norm, y_train)

# 점수를 출력
train_score = model.score(X_train_norm, y_train)
test_score = model.score(X_test_norm, y_test)
print(train_score, test_score)

# fit() 결과를 등고선으로 나타낸다
xmax = X_train_norm[:,0].max()+1
xmin = X_train_norm[:,0].min()-1
ymax = X_train_norm[:,1].max()+1
ymin = X_train_norm[:,1].min()-1

xx=np.linspace(xmin,xmax,200)
yy=np.linspace(ymin,ymax,200)
data1, data2 = np.meshgrid(xx,yy)
X_grid = np.c_[data1.ravel(), data2.ravel()]
decision_values = model.predict_proba(X_grid)[:,0] # 등고선을 위해 확률점수를 구함

sv=model.support_vectors_

fig=plt.figure(figsize=[14,12])

# show probability countour
CS=plt.contour(data1,data2,decision_values.reshape(data1.shape),levels=[0.01, 0.1, 0.5, 0.9, 0.99])#contourf는 색칠도 해준다.
plt.clabel(CS, inline=2, fontsize=10)

# show support vectors
plt.scatter(sv[:,0], sv[:,1], marker='s', c= 'k', s=100) # k는 검은색

# show train samples
plt.scatter(X_train_norm[:,0][y_train==0],X_train_norm[:,1][y_train==0],marker='o',c='r',label='malignant')
plt.scatter(X_train_norm[:,0][y_train==1],X_train_norm[:,1][y_train==1],marker='^',c='g',label='benign')

plt.legend()
plt.colorbar(CS,shrink=0.5)
plt.xlabel(cancer.feature_names[col1])
plt.ylabel(cancer.feature_names[col2])
plt.title('SVM - decision bounds',fontsize=20)
plt.show()