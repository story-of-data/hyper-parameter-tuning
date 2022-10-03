# Hyper Parameter Tuning using Random Forest

![Story-of-Data](https://user-images.githubusercontent.com/114144676/193519237-926fc194-adbc-43f6-a3e2-d954fc417c9b.png)

## The main parameters used by a Random Forest
- criterion = the function used to evaluate the quality of a split
- max_depth = maximum number of levels allowed in each tree
- max_features = maximum number of features considered when splitting a node.
- min_samples_leaf = minimum number of samples which can be stored in a tree leaf.
- min_samples_split = minimum number of samples necessary in a node to cause node splitting.
- n_estimarors = number of trees in the ensemble.

## GridSearchCV

```
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'],
                         rf_randomcv.best_params_['min_samples_leaf'] + 2,
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'],
                          rf_randomcv.best_params_['min_samples_split'] + 1,
                          rf_randomcv.best_params_['min_samples_leaf'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200,
                    rf_randomcv.best_params_['n_estimators'] - 100,
                    rf_randomcv.best_params_['n_estimators'],
                    rf_randomcv.best_params_['n_estimators'] + 100,
                    rf_randomcv.best_params_['n_estimators'] + 200,
                    rf_randomcv.best_params_['n_estimators'] - 600,]
}

```


## RandomizedSearchCV

```
from sklearn.model_selection import RandomizedSearchCV

# Create thre random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'criterion': ['entropy', 'gini']
}
```

## Automated Hyperparameter Tuning
 Automated Hyperparameter Tuning can be done by using techniques such as
- Bayesian Optimization
- Gradient Descent
- Evolutionary Algorithms

### Bayesian Optimization
- It uses the probability to find the minimum of a function. The final aim to find the input value of a funciton which can gives us the lowest output value. It usually performs better than random grid and manual search providing better performance in the testing phase and reduced optimization time. In Hyperopt, Bayesian Optimization can be implemented giving 3 main parameters to the function fmin.
- Objective Function = defines the loss function to minimize
- Domain Space = define the range of input value of test (in Bayesian Optimization this space creates a probability distribution for each of the used Hyperparameters)
- Optimization Algorithm = defines the search algorithm to use to select the best input values to use in each new iteration.


```
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# hp is used to define whether we are defining interger values, floating values, or choice function
space = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']),
    'max_depth': hp.quniform('max_depth', 10, 1200, 10),
    'max_featuers': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
    'min_samples_split': hp.uniform('min_samples_split', 0, 1),
    'n_estimators': hp.choice('n_estimators', [10, 50, 300, 750, 1200, 1300, 1500])
}

```


#### Thank you for reading
