import json
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


class XGBModelTuner:
    """
    A utility class for tuning XGBoost models using hyperparameter optimization.

    Attributes:
        model (xgb.XGBModel): An XGBoost model instance (e.g., XGBClassifier or XGBRegressor).
        param_grid (dict): Dictionary of hyperparameters to search over.
        optimizer (sklearn.model_selection.BaseSearchCV): 
            Search algorithm to use for optimization. Defaults to RandomizedSearchCV.
        X_train (array-like): Training feature data.
        Y_train (array-like): Training labels.
    """
    def __init__(self, model: xgb.XGBModel, param_grid: dict, X_train, Y_train, 
                 optimizer=RandomizedSearchCV):
        """
        Initializes the XGBModelTuner with a model, parameter grid, and training data.

        Args:
            model (xgb.XGBModel): The XGBoost model to be tuned.
            param_grid (dict): A dictionary defining the parameter search space.
            X_train (array-like): Training features.
            Y_train (array-like): Training labels.
            optimizer (class, optional): A search class such as RandomizedSearchCV or GridSearchCV.
                                         Defaults to RandomizedSearchCV.
        """
        self.model = model
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.X_train = X_train
        self.Y_train = Y_train

    def run(self, save_model=False, save_params=False, model_path='xgb_model.json', 
            param_path='model_params.json', cv=5, n_jobs=-1, verbose=2):
        """
        Runs the hyperparameter search using the specified optimizer.

        Args:
            save_model (bool, optional): Whether to save the best model to disk. Defaults to False.
            save_params (bool, optional): Whether to save the best parameters to disk. Defaults to False.
            model_path (str, optional): File path for saving the model. Defaults to 'xgb_model.json'.
            param_path (str, optional): File path for saving parameters. Defaults to 'model_params.json'.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1 (use all processors).
            verbose (int, optional): Verbosity level. Defaults to 2.

        Returns:
            xgb.XGBModel: The best fitted XGBoost model found by the search.
        """
        search = self.optimizer(estimator=self.model,
                               param_distributions=self.param_grid,
                               cv=cv,
                               n_jobs=n_jobs,
                               verbose=verbose)
        
        search.fit(self.X_train, self.Y_train)
        print(f"Best model: {search.best_params_} | Score: {search.best_score_}")
        if save_params:
            with open(param_path, 'w') as f:
                json.dump(search.best_params_, f)

        if save_model:
            search.best_estimator_.save_model(model_path)
        
        return search.best_estimator_