import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import KFold

"""
This module is a library of hyperparameters available for different 
learning algorithms. This library would be employed during a machine-learning pipeline
where an analyst would optimize hyperparmeters during cross-validation
"""

class HyperparameterOptimization(object):

    def __init__(self):
        
        
        self.extra_trees_hyp = {"max_depth": [4, 8, 12, 16, 20],
                                "min_samples_split": sp_randint(2, 11),
                                "min_samples_leaf": sp_randint(1, 11),
                                "n_estimators": np.arange(10,801,40)}

        self.gam_hyp = {'n_splines': np.arange(5,40)}
        


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
            
    def hyper_optimize(self, model, X, y, tscv = KFold(n_splits = 5), n_iter_search = 20):

        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(model, 
                                           cv = tscv, 
                                           param_distributions = self.extra_trees_hyp, 
                                           n_iter=n_iter_search)
        random_search.fit(X, y)
        
        best_ind = np.argmax(random_search.cv_results_['mean_test_score'])
        
        opt_hyp = random_search.cv_results_['params'][best_ind]
        
        self.hyper_report(random_search.cv_results_, n_iter_search)
        
        return opt_hyp