import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os, sys
# import accurecy_library for scoring attributes

from sklearn.metrics import accuracy_score
path = os.path.abspath(os.path.split(sys.argv[0])[0])
# importing the dataset
df = pd.read_csv(path+'/testsample6.csv')

# filling null value as 0
df = df.fillna('0')

# Dropping Age column
df = df.drop(['Age'], axis=1)
df = df.drop(['Age2'], axis=1)
# Dropping Perm_Address column
df = df.drop(['Perm_Address'], axis=1)

# taking independent variable
X = df.iloc[:, 1:22].values
# taking dependent variable
y = df.iloc[:, -1].values


labelencoder_X = LabelEncoder()
for i in range(0, 21):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

for i in range(1, 50):
    # split data train 90 % and test 10 %
    from sklearn.model_selection import train_test_split

    # X_train=X,y_train=y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)

    # import random forest classifier
    from sklearn.ensemble import RandomForestClassifier

    random_state = np.random.RandomState(0)
    # taking different peremater for random forest classifier
    clf = RandomForestClassifier(n_estimators=80, max_depth=15, max_features='auto', bootstrap='True',
                                 random_state=random_state, min_samples_leaf=4, oob_score=True)
    # fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with
    # a .predict() method call.
    y_score = clf.fit(X_train, y_train)
    # find out the predicted value
    predicted = clf.predict(X_test)
    # finding actual prediction compared with test size
    #print("Accuracy: ",
    #      accuracy_score(y_test, predicted, normalize=False))  # accuracy_score(train output, predict output)

    y__rf__score = clf.oob_decision_function_


    # print(y__rf__score)

# Learn to predict each class against the other
prediction_le = LabelEncoder().fit_transform(predicted)
# print(prediction_le)

y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                              25, 26, 27])

n_classes = y.shape[1]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(27):
    print(y[:, i])
    print(y__rf__score[:, i])
for i in range(27):
    curvy = roc_curve(y[:, i], y__rf__score[:, i])
    print(curvy)
    fpr[i], tpr[i] = curvy
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
