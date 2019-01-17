import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from pygam import GAM
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import r2_score

"""
This module is a library of hyperparameters available for different 
learning algorithms. This library would be employed during a machine-learning pipeline
where an analyst would optimize hyperparmeters during cross-validation
"""

class HyperparameterOptimization(object):

    def __init__(self, algorithm, params = None):
        
        self.algorithms = {'etr': (ExtraTreesRegressor(), {"max_depth": [4, 8, 12, 16, 20],
                                                           "min_samples_split": np.arange(2, 11),
                                                           "min_samples_leaf": np.arange(1, 11),
                                                           "n_estimators": np.arange(10,801,40)}),
                           'gam': (GAM(), {'n_splines': np.arange(5,40)})
                           }
        
        self.alg_selection = self.algorithms[algorithm][0]
        
        if params is None:
            self.alg_hyper = self.algorithms[algorithm][1]
        else:
            self.alg_hyper = params
        
        self.my_scorer = make_scorer(r2_score, greater_is_better = True)


    def hyper_report(self, results, n_top = 5):
        '''
        Output hyperparameter optimization results
        '''
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print "Model with rank: {0}".format(i) + '\n'
                print "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]) + '\n'
                print "Parameters: {0}".format(results['params'][candidate]) + '\n'
                print ""
            
    def hyper_optimize(self, model, X, y, cv = KFold(n_splits = 5), n_iter_search = 20):

        from sklearn.model_selection import RandomizedSearchCV
        
        self.random_search = RandomizedSearchCV(model, 
                                           cv = cv, 
                                           param_distributions = self.alg_hyper, 
                                           n_iter=n_iter_search,
                                           scoring = self.my_scorer,
                                           verbose = 1)
        self.random_search.fit(X, y)        
        self.opt_hyp = self.random_search.best_params_        
        self.hyper_report(self.random_search.cv_results_, n_iter_search)
        
        return self.random_search.predict
        