import numpy as np
from sklearn.model_selection import KFold

from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import r2_score

"""
This module is a library of machine learning algorithms and associated 
hyperparameter ranges suitable for wind energy analysis. This module allows
for simple implementation of hyperparameter optimization and the application of
the best hyperparameter combinations for use in the predictive model. 

For the most part, this toolkit is effectively a wrapper over SciKit-Learn machine
learning algorithms that help ensure consistent and appropriate use of the 
machine learning algorithms. Specifically, this toolkit constrains the following:
    
    1. The learning algorithms available 
    
    These include extremely randomized trees, gradient boosting, a generalized 
    additive model (from pyGAM library, not SciKit-Learn) and a support vector 
    machine. These algorithms have been used extensively at NREL and have been
    found to provide robust results for a range of turbine and wind plant power
    analyses.
    
    2. Hyperparameter ranges for optimizing algorithms
    
    Specific hyperparameter ranges for each algorithm are provided in this
    toolkit. Similar to the learning algorithms, the hyperparameter ranges
    are based on NREL's experience in applying these algorithms to several
    turbine and wind plant energy analyses. 
    
    3. The type of grid search used for cross-validation
    
    Two main types of grid searches are generally used: 
        exhaustive: in which all possible combinations of hyperparameters are 
                     considered, and:
        randomized: in which random combinations are chosen and capped at a 
                    specified level
        
    In this toolkit, we implement randomized grid search only, due to the number 
    of hyperparameters and the magnitude of ranges for certain algorithms. We set 
    as default 20 randomized samples, although this can be customized within the call 
    of the hyper_optimize function.
    
    4. The train-test split
    
    Hyperparameter optimization is performed through cross-validation of the 
    feature and response data. By default, we assume k-fold cross-validation (i.e. 
    data is randomly partitioned into 'k' equal-sized subsamples without reordering).
    The default 'k' value used in this toolkit is 5 (i.e. 80%/20% train-test split).
    Both the type of cross-validation and 'k' value can be customized within the call
    of the hyper_optimize function.
    
    5. Model performance score
    
    When optimzing hyperparamters, model performance is assessed based on the 
    coefficient of determination, or R2. The scorer can also be customized (e.g. RMSE)
    within the call of the hyper_optimize function.
       
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
            self.algorithm = ExtraTreesRegressor()
        
        elif algorithm == 'gbm': # Gradient boosting model
            from sklearn.ensemble import GradientBoostingRegressor
            self.hyper_range = {"max_depth": [4, 8, 12, 16, 20],
                                "min_samples_split": np.arange(2, 11),
                                "min_samples_leaf": np.arange(1, 11),
                                "n_estimators": np.arange(10,801,40)}
            self.algorithm = GradientBoostingRegressor()
        
        elif algorithm == 'gam': # Generalized additive model
            from pygam import GAM
            self.hyper_range = {'n_splines': np.arange(5,40)}
            self.algorithm = GAM()
        
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
        