"""
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>
.. codeauthor:: Mirko Polato

================
Kernel functions
================

.. currentmodule:: MKL.metrics.pairwise

This module contains all kernel functions of MKL. These kernels can be classified in
tho classes:
* kernels used in learning phases, such as HPK as SSK;
* kernels used in evaluation, such as identity and ideal kernel.

"""

import numpy as np
import cvxopt as co #TODO: remove this, using only numpy
from scipy.special import binom
from scipy.sparse import issparse
from MKL.utils.validation import check_X_T
import types

def homogeneous_polynomial_kernel(X, T=None, degree=2):
    """performs the HPK kernel between the samples matricex *X* and *T*.
    The HPK kernel is defines as:
    .. math:: k(x,z) = \langle x,z \rangle^d

    Parameters
    ----------
    X : (n,m) array_like,
        the train samples matrix.
    T : (l,m) array_like,
        the test samples matrix. If it is not defined, then the kernel is calculated
        between *X* and *X*.

    Returns
    -------
    K : (l,n) ndarray,
        the HPK kernel matrix.
    """
    
    X, T = check_X_T(X, T)
    return np.dot(X,T.T)**degree


#----------BOOLEAN KERNELS----------

def monotone_conjunctive_kernel(X,T=None,c=2):
    X, T = check_X_T(X, T)
    L = np.dot(X,T.T)
    return binom(L,c)

def monotone_disjunctive_kernel(X,T=None,d=2):
    X, T = check_X_T(X, T)
    L = np.dot(X,T.T)
    n = X.shape[1]

    XX = np.dot(X.sum(axis=1).reshape(X.shape[0],1), np.ones((1,T.shape[0])))
    TT = np.dot(T.sum(axis=1).reshape(T.shape[0],1), np.ones((1,X.shape[0])))
    N_x = n - XX
    N_t = n - TT
    N_xz = N_x - TT.T + L

    N_d = binom(n, d)
    N_x = binom(N_x,d)
    N_t = binom(N_t,d)
    N_xz = binom(N_xz,d)
    return (N_d - N_x - N_t.T + N_xz)


def monotone_dnf_kernel(X,T=None,d=2,c=2):
    X, T = check_X_T(X, T)
    n = X.shape[1]
    n_c = binom(n,c)
    XX = np.dot(X.sum(axis=1).reshape(X.shape[0],1), np.ones((1,T.shape[0])))
    TT = np.dot(T.sum(axis=1).reshape(T.shape[0],1), np.ones((1,X.shape[0])))
    XXc = binom(XX,c)
    TTc = binom(TT,c)
    return binom(n_c,d) - binom(n_c - XXc, d) - binom(n_c - TTc.T, d) + binom(my_mdk(X,T,c),d)


def monotone_cnf_kernel(X,T=None,c=2,d=2):
    pass

def conjunctive_kernel(X,T=None,c=2):
    pass

def disjunctive_kernel(X,T=None,d=2):
    pass

def dnf_kernel(X,T=None,d=2,c=2):
    pass

def cnf_kernel(X,T=None,c=2,d=2):
    pass

def tanimoto_kernel(X,T=None):#?
    X, T = check_X_T(X, T)
    L = np.dot(X,T.T)
    xx = np.linalg.norm(X,axis=1)
    tt = np.linalg.norm(T,axis=1)
