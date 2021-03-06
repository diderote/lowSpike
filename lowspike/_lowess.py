'''
Python implementation of a Lowess transformer based on non-parametric
    local regression models of inter-sample deltas.

'''

# Author: Daniel L Karl
# License: MIT

import itertools
import numbers

import numpy as np

from statsmodels.nonparametric import smoothers_lowess

__all__ = ['lowessNorm']


def _max_diff(M, combinations):
    '''
    Find the sample combination with the greatest average signal difference.
    
    Parameters
    ----------
    M : array-like shape(n_features, n_samples)
        Matrix of values
    
    combinations: array shape((n_samples choose 2), 2)
        Array of unique pairwise sample combinations
        
    Returns
    -------
    max_d: array shape(2)
        Indicies of sample combinations with largest mean value difference
    
    '''
    
    mx = 0
    
    for s1,s2 in combinations:
        c = abs((M[:,s1] - M[:,s2]).mean())
        if c > mx:
            max_d = np.array([s1, s2])
            mx = c
    return max_d


def _squared(x):
    return x**2


def _choose_idx_pmf(M, fun, random_state, size=50000):
    '''
    Chooses indicies based on a probability mass function.
    
    Parameters
    ----------
    M: array shape(n_features, 2)

    fun: callable
        Function to estimate a probability mass function

    size:  int
        Number of indicies to return
    
    Returns
    -------
    array shape(size)
        Vector of chosen indicies.
    '''
    
    difference = M[:,0] - M[:,1]

    unique_idx = np.unique(difference, return_index=True)[1]

    if len(unique_idx) <= size:
        return unique_idx

    else:
        d2 = fun(difference)
        total = d2.sum()
        pmf2 = d2 / total
        
        return random_state.choice(pmf2.shape[0], size=size, replace=False, p=pmf2)


def _get_subset_indicies(M, func, combinations, random_state, size=50000):
    '''
    From target values, choose indicies for modeling based on 
    the samples with the greatest mean signal difference.
    
    Parameters
    ----------
    M: array shape(target_features, n_samples)
        Matrix of values from which to subselect indicies for modeling.
  
    fun: callable
        Function for determining a PMF for subset selection
    
    combinations: array shape((n_samples choose 2), 2)
        Array of unique sample combinations
    
    size: int
        Number of indicies to return
    
    Returns
    -------
    array shape(size)
        Vector of chosen indicies.
    
    '''
    
    C = _max_diff(M, combinations)
    idx = _choose_idx_pmf(M[:,C], func, size=size, random_state=random_state)
    
    # maximize number of useful indicies if small number of indicies found using _max_diff
    if len(idx) < size:
        for comb in combinations:
            n_idx = _choose_idx_pmf(M[:,comb], func, size=size, random_state=random_state)
            idx = np.unique(np.concatenate((idx, n_idx)))
    
    return idx


def _delta(D, x, y, low, s, extrap_fraction, transform=False):
    '''
    Interpolates the lowess model to predict correction values.
    
    Parameters
    ----------
    D: array shape(len(s_i))
        Vector of calucated differences between two samples from target data
        
    x: array shape(n_features)
        Training Values

    y: array shape(len(s_i)) 
        Target values
    
    low: array shape(subset_size, 2) or 0
        Sample fit x and y of lowess model for sample
    
    s: float
        Learning Rate

    extrap_fraction: float <0,1)
        Fraction of largest y signal value to use for linear extrapolation when x > y.
        If 0, no extrapolation and max(y) used for x > y.
        
    transform: bool (default, False)
        Apply model to training data
    
    Returns
    -------
    ddx: array shape(n_featurtes)
        Vector of correction values for one sample
    
    ddy: array shape(len(s_i))
        Vector of correction values for one sample target data
    
    '''

    # linear at high signal
    low = np.unique(low, axis=0)

    # transform y
    y_interp = s * np.interp(y, low[:,0], low[:,1])

    if transform is False:
        return (0, y_interp)

    x_interp = s * np.interp(x, low[:,0], low[:,1])

    if extrap_fraction > 0:
        top = round(len(low) * extrap_fraction)
        
        #least squares linear extrapolation
        A = np.vstack([low[-top:,0], np.ones(top)]).T
        m, c = np.linalg.lstsq(A, low[-top:,1], rcond=None)[0]

        #replace x > y.max() with linear extrapolated values
        ext_idx = np.argwhere(x > low[:,0].max())
        x_interp[ext_idx] = m * x[ext_idx] + c

    return (x_interp, y_interp)


def _check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Taken from sklearn
    
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

    
def _delta_matrix(X, y, s_i, combinations, learning_rate, extrap_fraction, 
                  lower_bound, transform=False):
    '''
    One step based on lowess modeled delta matrix
    
    X: array shape(n_features, n_samples)
        Train data
    
    y: array shape(y_features, n_samples)
        Target data
    
    s_i: array shape(len(subset_size))
        Indicies of subelected target data
    
    combinations: array shape((n_samples choose 2), 2)
        Array of unique sample combinations
    
    learning_rate: float, (default: 1.)
        x / n_samples
    
    extrap_fraction: float, <0,1) 
        Fraction of largest y values for extrapolation when x > y

    lower_bound : int, float or None (default = None)
        Lower bound on transformed values

    transform: bool (default: False)
        Apply model delta to training data
        
    
    Returns
    -------
    X: array shape(n_features, n_samples)
        Lowess adjusted signal values
    
    y: array shape(y_features, n_samples)
        Lowess adjusted model
    
    se: array shape(n_samples)
        Squared Error vector

    '''
    
    dx, dy = np.zeros(X.shape), np.zeros(y.shape) #Initialize delta matrix
    
    for s1, s2 in combinations:
        #Calculated differences betwee two samples
        D = y[s_i, s1] - y[s_i, s2]
        
        for s_ in [s1,s2]:
            #Model the deltas samples based on y signal.
            low = smoothers_lowess.lowess(D, y[s_i, s_])[:,:2]
            #Apply the correction
            ddx, ddy = _delta(D, X[:,s_], y[:, s_], low, learning_rate, extrap_fraction, transform)
            dx[:,s_], dy[:,s_] = dx[:,s_] - ddx, dy[:,s_] - ddy
    
    X, y = X - dx, y - dy
    se = dy**2

    if isinstance(lower_bound, (int, float)):
        X, y = X.clip(lower_bound), y.clip(lower_bound)

    return X, y, se
        
        
def normalize_lowess(X, y=None, subset_size=50000, sample_weight=None, max_iter=5, 
                      tol=1e-3, learning_rate=1., fun='squared', fun_args=None, 
                      random_state=None, extrap_fraction=.1, lower_bound=None,
                      transform=False):
    '''
    Performs a lowess normalization and transformation based on non-parametric
    local regression model of sample differences.
    
    Executes an iterative optimization of target features deltas based
    on a non-parametric local regression model and applies a transformer.
    
    Parameters
    ----------
    X: array shape(n_features, n_samples)
        Train data
    
    y: array shape(y_features, n_samples) or None
        Target data (ie. spike)
    
    subset_size : int, optional
        Number of regions to guide the transformation.  If None, all are used.
    
    sample_weight : array-like of shape (n_samples,) (default=None)
        Individual weights for each sample.  If None, every sample
        will have the same weight.
    
    max_iter : int, optional
        Maximum nubmer of iterations during fit.
        
    tol : float, (default: 0.001)
        A positive scalar giving the tolerance at which the
        model is considered to have converged. 
    
    learning_rate : float, (default: 1.)
        Learning rate at each iteration. Multiple of the per sample step at each 
        iteration.  ie. Learning rate of 1.0 corrects by 1.0 / num_samples at each 
        iteration.

    fun : string or function, optional. Default: 'squared'
        Function used for determining the weigthed probability for subset selection.
    
    fun_args : dictionary, optional
        Arguements to send to the function.
        If empty and if fun='squared', empty dictionary will be passed.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    extrap_fraction : float <0,1) default: 0.1
        Fraction of largest y values to use for extrapolation where x > y.

    lower_bound : int, float or None (default = None)
        Lower bound on transformed values

    transform : bool, (default=False)
        Whether to correct traning values
    
    Returns
    -------
    X_new: array, shape(n_samples, n_features)
        Transformed data matrix
    
    y_new: array, shape(n_samples, subset_size)
        Transformed subset data matrix
    
    mse: float
        Mean square error at break
    
    n_iter: int
        The maximum number of iterations taken to converge or break.
    
    squared_errors: array, shape(max_iter, n_samples)
        Squared errors at each iteration
    
    s_comb: array, shape((n_samples choose 2), 2)
        The combinations of samples in order of comparison for lowess.
    
    '''
    
    random_state = _check_random_state(random_state)
    fun_args = {} if fun_args is None else fun_args
    
    if fun == 'squared':
        f = _squared
    elif callable(fun):
        def f(x, fun_args):
            return fun(x, **fun_args)
    else:
        exc = ValueError if isinstance(fun, str) else TypeError
        raise exc("Unknown function %r;"
                  " should be one of 'squared' or callable"
                  % fun)

    n_samples, n_features = X.shape
    lr = learning_rate / (n_samples)
    
    X_new = X.T.copy()

    #get pairwise combinations of samples
    s_comb = np.array(list(itertools.combinations(range(n_samples),2)))

    del(X)

    if y is None:
        y_idx = _get_subset_indicies(X_new, f, s_comb, random_state, size=subset_size)
        y_new = X_new[y_idx,:]
    else:
        y_new = y.copy()

    del(y_idx)
    
    for ii in range(max_iter):

        s_i = _get_subset_indicies(y_new, f, s_comb, random_state, size=subset_size)
        X_new, y_new, se = _delta_matrix(X_new, y_new, s_i, s_comb, lr, extrap_fraction, 
                                         lower_bound, transform)

        mse = se.mean()
        if mse < tol:
            break
    else:
        print('Model did not converge. Consider increasing ' + 
              'tolerance or the maximum number of iterations.')
    
    # capture se matrix?
    X_new = X_new.T
    y_new = y_new.T
    
    return X_new, y_new, mse, ii + 1, s_comb

    
class lowessNorm():
    '''
    A method to guide sample normalizaiton based on lowess model of sample differences.
    
    Parameters
    ----------
    subset_size : int, optional
        Number of regions to guide the transformation.  If None, all are used.
    
    sample_weight : array-like of shape (n_samples,) (default=None)
        Individual weights for each sample.  If None, every sample
        will have the same weight.
    
    max_iter : int, optional
        Maximum nubmer of iterations during fit.
        
    tol : float, (default: 0.001)
        A positive scalar giving the tolerance at which the
        model is considered to have converged. 
    
    learning_rate : float, (default: 1.)
        Learning rate at each iteration. Multiple of the per sample step at each 
        iteration.  ie. Learning rate of 1.0 corrects by 1.0 / num_samples at each 
        iteration.

    fun : string or function, optional. Default: 'squared'
        Function used for determining the weigthed probability for subset selection.
    
    fun_args : dictionary, optional
        Arguements to send to the function.
        If empty and if fun='squared', empty dictionary will be passed.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    extrap_fraction : float (0,1) (default=.1)
        Fraction of largest y values to use for extrapolation where x > y.
        0 -> no extrapolation and max(y) used for x > y

    lower_bound : int, float or None (default = None)
        Lower bound on transformed values

        
    Attributes
    ----------
    
    X_new_: array, shape(n_samples, n_features)
        Transformed data matrix
    
    y_new_: array, shape(n_samples, subset_size)
        Transformed subset data matrix
    
    mse_: float
        Mean square error at convergence or max_iter
    
    n_iter_: int
        The maximum number of iterations taken to converge or break.
    
    squared_errors_: array shape(max_iter, n_samples)
        Squared errors at each iteration until convergence or max_iter
    
    sample_combinations_: array, shape((n_samples choose 2), 2)
        The combinations of samples in order of comparison for lowess.
    
    '''
    def __init__(self, subset_size=50000, sample_weight=None, max_iter=5, 
                 tol=1e-3, learning_rate=1., fun='squared', fun_args=None, 
                 random_state=None, extrap_fraction=0.1, lower_bound = None):
        self.subset_size = subset_size
        self.sample_weight = sample_weight
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.fun = fun
        self.fun_args = fun_args
        self.random_state = random_state
        self.extrap_fraction = extrap_fraction
        self.lower_bound = lower_bound
    
    
    def _fit(self, X, y=None, transform=False):
        '''
        Fit the model
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data
        
        y: array-like, shape(n_samples, n_features), or None.
            Target values to guide normlization.  If None,
            targets will be selected from training data.
        
        tranform: bool
            If False, compute the transformation only for the target y.

        
        Returns
        -------
            X_new: array-like, shape (n_samples, n_features)
            
        '''
    
        fun_args = {} if self.fun_args is None else self.fun_args
    
        X_new, y_new, mse, n_iter, s_comb  = normalize_lowess(X=X, y=y, 
            subset_size=self.subset_size, sample_weight=self.sample_weight,
            max_iter=self.max_iter, tol=self.tol, learning_rate=self.learning_rate,
            fun=self.fun, fun_args=fun_args, random_state=self.random_state,
            extrap_fraction=self.extrap_fraction, lower_bound=self.lower_bound,
            transform=transform)
    
        self.y_new_ = y_new
        self.mse_ = mse
        self.n_iter = n_iter
        self.sample_combinations_ = s_comb
        
        if transform:        
            self.X_new = X_new
        
            return X_new
    
    
    def fit_transform(self, X, y=None):
        '''
        Fit the model and transform the data
        
        Parameters
        ----------
        X : array-like shape(n_bins, n_samples)
            Data 
        
        y : array-like 
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
        
        '''
        
        return self._fit(X, y, transform=True)
    
    
    def fit(self, X, y=None):
        '''
        Fit the model
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data
        
        y: array-like, shape(n_samples, n_features), or None.
            Target values to guide normlization.  If None,
            targets will be selected from training data.
        
        Returns
        -------
        self
        '''
        
        self._fit(X, y=None, transform=False)
        return self
    
    
    #def transform(self, copy=True):
        
    
        # check is fitted
        # perform iterative changes on X
        # maybe change so if None, subset == fit y
    
    #def predict() ??