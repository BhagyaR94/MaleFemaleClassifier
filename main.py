import os
import pickle

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

input_dir = 'E:/Tutorials/Python/MaleFemaleClassifier/data_light'
categories = ['male', 'female']
data = []
labels = []

for cat_idx, cat in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, cat)):
        img_path = os.path.join(input_dir, cat, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(cat_idx)
print('image reading completed!')
data = np.asarray(data)
labels = np.asarray(labels)
print('size of data', len(data))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)
print('split completed!')

classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
print('training')
grid_search.fit(x_train, y_train)
print('training completed!')

best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print('Score: ', score * 100)
pickle.dump(best_estimator, open('./model.p', 'wb'))
