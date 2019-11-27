'''
Python implementation of a Lowess transformer based on non-parametric
    local regression models of inter-sample deltas.

'''

# Author: Daniel L Karl
# License: MIT clause

import itertools
import numbers

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from statsmodels.nonparametric import smoothers_lowess

__all__ = ['lowessTransform']


def _max_diff(df, combinations):
    '''
    Find the sample combination with the greatest average signal difference.
    
    Parameters
    ----------
    df : array-like of shape = [n_bins, n_samples]
    combinations: list of non-repeating sample lists [(s1,s2), (s1,s3), (s2,s3)]
    
    Returns
    -------
    list of samples with greatest mean signal differences
    '''
    
    mx = 0
    comb = ['None', 'None']
    
    for s1,s2 in combinations:
        comb_max = abs((df[s1] - df[s2]).mean())
        if comb_max > mx:
            comb = [s1, s2]
            mx = comb_max
    return comb


def _squared(x)
    return x**2


def _choice_idx_pmf(df, fun, size=50000):
    '''
    Makes a pmf based on signal intensity and chooses random indicies based on pmf
    
    Parameters
    ----------
    df: dataframe with two columns
    size:  how many indicies to choose
    
    Returns
    -------
    Values of indicies from the df based on the pmf
    '''
    
    difference = df.iloc[:,0] - df.iloc[:,1]
    d2 = fun(difference)
    total = d2.sum()
    pmf2 = d2 / total
    
    return np.random.choice(pmf2.index, size=size, replace=False, p=pmf2.values)


def _get_subset_indicies(subset_df, combinations, size=50000):
    '''
    From spike dataframe, choose inidicies for modeling based on 
    the samples with the greatest mean signal difference.
    
    Parameters
    ----------
    df: dataframe of spike or subset signal values
    combinations: list of non-repeating sample lists [(s1,s2), (s1,s3), (s2,s3)]
    size: number of indicies to return
    
    Returns
    -------
    list-like of chosen indicies 
    
    '''
    
    comb = max_diff(subset_df, combinations)
    return choice_idx_pmf(subset_df[comb], size=size)


def _delta(difference, values, subset_indicies, num_samples):
    '''
    Makes a lowess model of the difference between samples based on the signal value 
    of the spike indicies (genomic bin) in that sample.  Calcualte the differences over 
    the entire dataframe.  (ie. Based on signal value at each index, calculate the 
    adjustment of the signal at that index in the direction that the spike in is 
    different to the other sample.)
    
    Parameters
    ----------
    difference: vector of calucated differences between two samples at spike-in indicies
    values: signal values of one sample at spike-in indicies
    num_samples: number of total samples in the experiment
    
    Returns
    -------
    list of the correction values for that sample compared to one other sample
    '''
    
    low = smoothers_lowess.lowess(difference, values.loc[subset_indicies])
    interp = interp1d(low[:,0], low[:,1], bounds_error=False, 
                      fill_value=(difference[values.loc[subset_indicies].idxmin()], 
                                  "extrapolate")) # linear at high signal 
    return interp(values) / (num_samples - 1)


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


def _normalize_lowess(df, spike_indicies, subset_size=50000, iterations=5, logit=True):
    '''
    Executes a minimization of the differences between spike-in samples based
    on a non-parametric local regression model and applies the corrections to 
    all datapoints.
    
    Parameters
    ----------
    df: dataframe with sample names in columns. (binned genome signal means of IP and spike-in)
    spike_indicies: list of spike-in indicies
    subset_size: number of indicies to build model
    iterations: number of iterations
    logit: apply a log1p to the dataframe
    
    Returns
    -------
    Normalized Dataframe
    Dictionary of the max mean squared error at each iteration
    '''
    
    errors={}
    
    d = df.apply(np.log1p).copy() if logit else df.copy()
    
    samples = d.columns.tolist()
    combinations = list(itertools.combinations(samples, 2))
    
    for i in tqdm.tqdm(range(interations)):
        #Initialize a change matrix
        ddf = pd.DataFrame(0, columns=df.columns, index=d.index)
        #choose spike in indicies for modeling
        subset_index = get_subset_indicies(d.loc[spike_indicies], combinations, size=subset_size)
        
        for s1, s2 in combinations:
            #Calculate differences between two samples
            difference = df.loc[subset_index,s1] - df.loc[subset_index,s2]
            #Model the differences based on binned values and adjust the change matrix per comparison
            ddf[s1] = ddf[s1] + delta(difference, df[s1], subset_index, len(samples))
            ddf[s2] = ddf[s2] - delta(difference, df[s2], subset_index, len(samples))
            
        #Make the iteration adjustments to the entire dataset
        d = d - ddf
        
        #errors[f'{i + 1}'] = ddf.loc[sub_index, samples].mean()
        errors[f'{i + 1} MSE'] = ((ddf.loc[sub_index, samples])**2).mean()
    
    MSE = {k: df.max() for k,df in errors.items() if 'MSE' in k}
    
    normed_df = d.apply(np.expm1) if logit else d
    
    return normed_df, MSE


def normalize_lowess(subset_size=50000, sample_weight=None, max_iter=5, 
                      tol=1e-3, learning_rate=1., fun='squared', fun_args=None, 
                      random_state=None, return_n_iter=True, logit=True):
    '''
    Performs a lowess normalization and transoformation based on non-parametric
    local regression model of sample differences.
    
    Executes an iterative optimization of target features deltas based
    on a non-parametric local regression model and applies a transformer.
    
    Parameters
    ----------
    subset_size : int, optional
        Number of regions to guide the transformation.  If None, all are used.
    
    sample_weight: array-like of shape (n_samples,) (default=None)
        Individual weights for each sample.  If None, every sample
        will have the same weight.
    
    max_iter: int, optional
        Maximum nubmer of iterations during fit.
        
    tol : float, (default: 0.001)
        A positive scalar giving the tolerance at which the
        model is considered to have converged. 
    
    learning_rate: float, (default: 1.)
        Learning rate at each iteration. Multiple of the per sample step at each 
        iteration.  ie. Learning rate of 1.0 corrects by 1.0 / num_samples at each 
        iteration.

    fun: string or function, optional. Default: 'squared'
        Function used for determining the weigthed probability for subset selection.
    
    fun_args : dictionary, optional
        Arguements to send to the function.
        If empty and if fun='squared', empty dictionary will be passed.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    return_n_iter: bool, optional
        Whether or not to return the number of iterations.
        
    logit: bool, (default=True)
        Whether to log1p transform data
    
    Returns
    -------
    X_new_: array, shape(n_samples, n_features)
        Transformed data matrix
    
    y_new_: array, shape(n_samples, subset_size)
        Transformed subset data matrix
    
    mse_: array, shape(n_iter, n_samples)
        Mean square error per sample at each iteration
    
    n_iter_ : int
        The maximum number of iterations taken to converge or break.
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

    n, p = X.shape

    ## perform subset choice
    ## perform iteration
    
    if return_n_iter:
        return X_new, y_new, mse, n_iter
    else:
        return X_new, y_new, mse
    
    
class lowessTransform():
    '''
    A method to guide sample normalizaiton based on lowess model of sample differences.
    
    Parameters
    ----------
    subset_size : int, optional
        Number of regions to guide the transformation.  If None, all are used.
    
    sample_weight: array-like of shape (n_samples,) (default=None)
        Individual weights for each sample.  If None, every sample
        will have the same weight.
    
    max_iter: int, optional
        Maximum nubmer of iterations during fit.
        
    tol : float, (default: 0.001)
        A positive scalar giving the tolerance at which the
        model is considered to have converged. 
    
    learning_rate: float, (default: 1.)
        Learning rate at each iteration. Multiple of the per sample step at each 
        iteration.  ie. Learning rate of 1.0 corrects by 1.0 / num_samples at each 
        iteration.

    fun: string or function, optional. Default: 'squared'
        Function used for determining the weigthed probability for subset selection.
    
    fun_args : dictionary, optional
        Arguements to send to the function.
        If empty and if fun='squared', empty dictionary will be passed.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    return_n_iter: bool, optional
        Whether or not to return the number of iterations.
        
    logit: bool, (default=True)
        Whether to log1p transform data
        
    Attributes
    ----------
    
    X_new_: array, shape(n_samples, n_features)
        Transformed data matrix
    
    y_new_: array, shape(n_samples, subset_size)
        Transformed subset data matrix
    
    mse_: array, shape(n_iter, n_samples)
        Mean square error per sample at each iteration
    
    n_iter_ : int
        The maximum number of iterations taken to converge or break.
    
    '''
    def __init__(self, subset_size=50000, sample_weight=None, max_iter=5, 
                 tol=1e-3, learning_rate=1., fun='squared', fun_args=None, 
                 random_state=None, return_n_iter=True, logit=True):
        self.subset_size = subset_size
        self.sample_weight = sample_weight
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.fun = fun
        self.fun_args = fun_args
        self.random_state = random_state,
        self.return_n_iter = return_n_iter,
        self.logit = logit
    
    
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
            X_new : array-like, shape (n_samples, n_features)
        '''
    
        fun_args = {} if self.fun_args is None else self.fun_args
    
        X_new, y_new, mse, self.n_iter_ = normalize_lowess(X=X,, y=y, 
            subset_size=self.subset_size, sample_weight=self.sample_weight,
            max_iter=self.max_iter, tol=self.tol, learning_rate=self.learning_rate,
            fun=self.fun, fun_args=fun_args, random_state=self.random_state, 
            return_n_iter=self.return_n_iter, logit=self.logit, transform=transform)
    
        self.y_new_ = y_new
        self.mse_ = mse
        
        ## Maybe generate a model of the y step per iteration?
        ## So transform can apply this to X
        
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
    
    
    def transform(self, copy=True):
        
    
        # check is fitted
        # perform iterative changes on X
        # maybe change so if None, subset == fit y