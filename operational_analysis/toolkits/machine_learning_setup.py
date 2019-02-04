import numpy as np
from sklearn.model_selection import KFold

from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import r2_score

"""
This module is a library of machine learning algorithms and associated 
hyperparameter ranges suitable for wind energy analysis. This module allows
for simple implementation of hyperparameter optimization and the application of
the best hyperparameter combinations for use in the predictive model. 
"""

class MachineLearningSetup(object):

    def __init__(self, algorithm, params = None):
        '''
        Initialize the class with a list of possible algorithms and 
        recommended hyperparameter ranges
        '''    
        if algorithm == 'etr': # Extra trees regressor
            from sklearn.ensemble import ExtraTreesRegressor
            self.hyper_range = {"max_depth": [4, 8, 12, 16, 20],
                                "min_samples_split": np.arange(2, 11),
                                "min_samples_leaf": np.arange(1, 11),
                                "n_estimators": np.arange(10,801,40)}
            self.algorithm = ExtraTreesRegressor(**self.hyper_range)
        
        elif algorithm == 'gbm': # Gradient boosting model
            from sklearn.ensemble import GradientBoostingRegressor
            self.hyper_range = {"max_depth": [4, 8, 12, 16, 20],
                                "min_samples_split": np.arange(2, 11),
                                "min_samples_leaf": np.arange(1, 11),
                                "n_estimators": np.arange(10,801,40)}
            self.algorithm = GradientBoostingRegressor(**self.hyper_range)
        
        elif algorithm == 'gam': # Generalized additive model
            from pygam import GAM
            self.hyper_range = {'n_splines': np.arange(5,40)}
            self.algorithm = GAM(**self.hyper_range)
        
        elif algorithm == 'svm': # Support vector machine
            from sklearn.svm import SVR
            self.hyper_range = {"C": [0.1, 1, 10, 50, 100],
                                "gamma": [0.01, 0.1, 1, 10],
                                "kernel": ['poly', 'rbf', 'sigmoid']}
            self.algorithm = SVR(**self.hyper_range)
        
        # Set scorer as R2
        self.my_scorer = make_scorer(r2_score, greater_is_better = True)

    def hyper_report(self, results, n_top = 5):
        '''
        Output hyperparameter optimization results into terminal window in order 
        of mean validation score.
        
        Args:
            results(:obj:'dict'): Dictionary containg cross-validation results
            n_top(:obj:`int`): The number of results to output
        
        Returns:
            (none)
        '''
        
        # Loop through cross validation results and output to terminal
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i) + '\n')
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]) + '\n')
                print("Parameters: {0}".format(results['params'][candidate]) + '\n')
                print("")
            
    def hyper_optimize(self, X, y, cv = KFold(n_splits = 5), n_iter_search = 20, report = True):
        '''
        Optimize hyperparameters through cross-validation 
        
        Args:
            X(:obj:'numpy array or pandas dataframe): The inputs or features
            Y(:obj:'numpy array or pandas series): The target or predictand
            cv(:obj:'sklearn.model_selection._split'): The train/test splitting method
            n_iter_search(:obj:'int'): The number of random hyperparmeter samples to use
            report(:obj:'boolean'): Indicator on whether to output a summary report 
                                    on optimization results
        
        Returns:
            (none)
        '''
        from sklearn.model_selection import RandomizedSearchCV
        
        # Setup randomized cross-validated grid search
        self.random_search = RandomizedSearchCV(self.algorithm,
                                                cv = cv, 
                                                param_distributions = self.hyper_range, 
                                                n_iter=n_iter_search,
                                                scoring = self.my_scorer,
                                                verbose = 1,
                                                return_train_score = True)
        # Fit the model to each combination of hyperparmeters
        self.random_search.fit(X, y)        
        
        # Assign optimal parameters to object
        self.opt_hyp = self.random_search.best_params_        
        
        # Output results to terminal
        if report:
            self.hyper_report(self.random_search.cv_results_, n_iter_search)
        