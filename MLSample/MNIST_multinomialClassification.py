import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int)

print(y[62000])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#훈련세트를 섞음 (비슷한 샘플이 연이어 나타나면 성능이 나빠짐)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

some_digit = X[62000]

# 다중분류 (OvA)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, random_state=42)
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

# OvA 점수
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

print(np.argmax(some_digit_scores))
print(sgd_clf.classes_)
print(sgd_clf.classes_[5])

# 다중분류 (OvO)
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=1000, random_state=42))
ovo_clf.fit(X_train, y_train)
print(ovo_clf.predict([some_digit]))
print(len(ovo_clf.estimators_))

# RandomForestClassifier (분류기 자체적으로 다중클래스 분류가 가능)
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

print(forest_clf.predict_proba([some_digit]))

# 분류기의 교차검증 평가
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# 에러분석
# 오차행렬 출력
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

#이미지 표현
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# 에러 비율 계산 ( 오차행렬의 각 값을 대응되는 클래스의 이미지 개수로 나눔)
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
