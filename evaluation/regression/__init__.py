__author__ = 'Junghwan Kim'
__copyright__ = 'Copyright 2016-2019 Junghwan Kim. All Rights Reserved.'
__version__ = '1.0.0'


from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


# Explained variance score
def evs(y_true, y_pred):
    print('Explained variance score:', round(explained_variance_score(y_true, y_pred), 2))


# Max error
def max(y_true, y_pred):
    print('Max error:', round(max_error(y_true, y_pred), 2))


# Mean absolute error
def mae(y_true, y_pred):
    print('Mean absolute error:', round(mean_absolute_error(y_true, y_pred), 2))


# Mean squared error
def mse(y_true, y_pred):
    print('Mean squared error:', round(mean_squared_error(y_true, y_pred), 2))


# Mean squared logarithmic error
def msle(y_true, y_pred):
    print('Mean squared logarithmic error:', round(mean_squared_log_error(y_true, y_pred), 2))


# Median absolute error
def mdae(y_true, y_pred):
    print('Median absolute error:', round(median_absolute_error(y_true, y_pred), 2))


# R² score
def r2(y_true, y_pred):
    print('R² score:', round(r2_score(y_true, y_pred), 2))

