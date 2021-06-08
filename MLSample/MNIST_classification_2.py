import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(int)

print(y[64000])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#훈련세트를 섞음 (비슷한 샘플이 연이어 나타나면 성능이 나빠짐)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#이진분류기
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, random_state=42)
sgd_clf.fit(X_train, y_train_5)

some_digit = X[64000]
print(sgd_clf.predict([some_digit]))

#교차검증
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

#오차행렬
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(y_train_pred)

from sklearn.metrics import confusion_matrix
result = confusion_matrix(y_train_5, y_train_pred)
print(result)

y_train_perfect_predictions = y_train_5
result = confusion_matrix(y_train_5, y_train_perfect_predictions)
print(result)

#정밀도와 재현율
from sklearn.metrics import precision_score, recall_score

m_precision_score = precision_score(y_train_5, y_train_pred)
print(m_precision_score)
m_recall_score = recall_score(y_train_5, y_train_pred)
print(m_recall_score)

#F1점수
from sklearn.metrics import f1_score
m_f1_score = f1_score(y_train_5, y_train_pred)
print(m_f1_score)

#적절한 임계값 설정 (결정임곗값에 대한 정밀도와 재현율 그래프)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5,cv=3,method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precisions", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="recalls", linewidth=2)
    plt.xlabel("threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#정밀도와 재현율의 비율 그래프
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("precisions", fontsize=16)
    plt.ylabel("recalls", fontsize=16)
    plt.axis([0, 1, 0, 1])
plot_precision_vs_recall(precisions, recalls)
plt.show()
