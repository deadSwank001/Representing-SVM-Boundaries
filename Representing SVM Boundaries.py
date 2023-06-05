# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:54:52 2023

@author: swank
"""

Selecting on univariate measures
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
Selector_f = SelectPercentile(f_regression, percentile=25)
Selector_f.fit(X, y)
for n,s in zip(boston.feature_names,Selector_f.scores_):
    print('F-score: %3.2f\t for feature %s ' % (s,n))
Using a greedy search
from sklearn.feature_selection import RFECV
selector = RFECV(estimator=regression, 
                 cv=10, 
                 scoring='neg_mean_squared_error')
selector.fit(X, y)
print("Optimal number of features : %d" 
      % selector.n_features_)
print(boston.feature_names[selector.support_])
Pumping up your hyper-parameters
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
Implementing a grid search
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, 
      weights='uniform', metric= 'minkowski', p=2)
grid = {'n_neighbors': range(1,11), 
        'weights': ['uniform', 'distance'], 'p': [1,2]}
print ('Number of tested models: %i' 
       % np.prod([len(grid[element]) for element in grid]))
score_metric = 'accuracy'
from sklearn.model_selection import cross_val_score
print('Baseline with default parameters: %.3f' 
      % np.mean(cross_val_score(classifier, X, y, 
                cv=10, scoring=score_metric, n_jobs=1)))
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(estimator=classifier, 
                      param_grid=grid, 
                      scoring=score_metric, 
                      n_jobs=1, 
                      refit=True, 
                      return_train_score=True, 
                      cv=10)
search.fit(X,y)
print('Best parameters: %s' % search.best_params_)
print('CV Accuracy of best parameters: %.3f' % 
      search.best_score_)
print(search.cv_results_)
from sklearn.model_selection import validation_curve
model = KNeighborsClassifier(weights='uniform', 
                             metric= 'minkowski', p=1)
train, test = validation_curve(model, X, y, 
                               param_name='n_neighbors', 
                               param_range=range(1, 11), 
                               cv=10, scoring='accuracy', 
                               n_jobs=1)
import matplotlib.pyplot as plt
mean_train  = np.mean(train,axis=1)
mean_test   = np.mean(test,axis=1)
plt.plot(range(1,11), mean_train,'ro--', label='Training')
plt.plot(range(1,11), mean_test,'bD-.', label='CV')
plt.grid()
plt.xlabel('Number of neighbors')
plt.ylabel('accuracy')
plt.legend(loc='upper right', numpoints= 1)
plt.show()
​
Trying a randomized search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=classifier, 
                    param_distributions=grid, n_iter=10, 
    scoring=score_metric, n_jobs=1, refit=True, cv=10, )
random_search.fit(X, y)
print('Best parameters: %s' % random_search.best_params_)
print('CV Accuracy of best parameters: %.3f' % 
      random_search.best_score_)