# -*- coding: latin-1 -*-
"""
@author: Michele Donini
@email: mdonini@math.unipd.it
 
EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini
 
Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from .base import MKL
from ..multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsRestMKLClassifier as ovaMKL
from ..utils.exceptions import BinaryProblemError
from .komd import KOMD
from ..lists import HPK_generator
 
from cvxopt import matrix, spdiag, solvers
import numpy as np

from MKL.arrange import summation
import MKL.utils.helpers as helpers
import MKL.utils.kernel_helpers as k_helpers
 
class SimpleMKL(BaseEstimator, ClassifierMixin, MKL):
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.
 
        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini
 
        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, estimator=KOMD(lam=0.1), lam=0.1, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=100, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.lam = lam

        
    def _arrange_kernel(self):
        # The number of kernels
        M = len(self.KL)

        #The number of examples
        n = len(self.Y)
        #The weights of each kernel
        #Initialized to 1/M
        d = np.ones(M) / M
        D = np.ones(M)

        #Just a placeholder for something that gets updated later
        dJ = '-20'

        #Stores all the individual kernel matrices
        kernel_matrices = self.KL
        #Creates y matrix for use in SVM later
        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        y_mat = np.outer(Y, Y)

        #Gets constraints for running SVM
        box_constraints = helpers.get_box_constraints(n, 1.0)

        #Gets starting value for SVM
        alpha0 = np.zeros(n)

        combined_kernel_matrix = summation(kernel_matrices, d)
        # combined_kernel_func = k_helpers.get_combined_kernel_function(kernel_functions, d)

        #Gets J, also calculates the optimal values for alpha
        alpha, J, info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha0, box_constraints)
        J *= -1

        #Gradient of J w.r.t d (weights)
        dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)
        iteration = 0
        #Loops until stopping criterion reached
        while (max(D) != 0 and not helpers.stopping_criterion(dJ, d, 0.01)):
            iteration += 1
            print("iteration and weights:", iteration, d)

            combined_kernel_matrix = summation(kernel_matrices, d)
            # combined_kernel_func = k_helpers.get_combined_kernel_function(kernel_functions, d)

            #Gets J, also calculates the optimal values for alpha
            alpha, J, info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha0, box_constraints)
            J *= -1

            #Gradient of J w.r.t d (weights)
            dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)
            
            #The index of the largest component of d
            mu = d.argmax()

            #Descent direction
            #Basically, we are calculating -1 * reduced gradient of J w.r.t d,
            #using the index of the largest component of d as our "mu"
            #in the reduced gradient calculation
            D = helpers.compute_descent_direction(d, dJ, mu)
            D = helpers.fix_precision_of_vector(D, 0)
            J_cross = 0
            d_cross = d.copy()
            D_cross = D.copy()

            sub_iteration = 0

            #Get maximum admissible step size in direction D
            while (J_cross < J):
                sub_iteration += 1
                d = d_cross.copy()
                D = D_cross.copy()

                print ('  J:', J, '| J_cross:', J_cross)
                print ('    d cross', d_cross)
                print ('    d cross sum', sum(d_cross))

                print ('    D cross', D_cross)
                print ('    D cross sum', sum(D_cross))

                combined_kernel_matrix = summation(kernel_matrices, d)
                alpha, J, info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha, box_constraints)
                J *= -1

                #Maximum admissible step size
                gamma_max = 123456

                #argmax of above
                v = -0.123456

                #Find gamma_max and v
                for m in range(M):
                    if D[m] < 0:
                        d_D_quotient = -1 * d[m] / D[m]
                        if d_D_quotient < gamma_max:
                            gamma_max = d_D_quotient
                            v = m

                d_cross = d + gamma_max * D

                #Not strictly necessary, but helps avoid precision errors
                if (v >= 0):
                    d_cross[v] = 0
                    D_cross[mu] = D[mu] + D[v]
                    D_cross[v] = 0

                d_cross = helpers.fix_precision_of_vector(d_cross, 1)
                D_cross = helpers.fix_precision_of_vector(D_cross, 0)

                combined_kernel_matrix_cross = summation(kernel_matrices, d_cross)
                alpha_cross, J_cross, cross_info = helpers.compute_J(combined_kernel_matrix_cross, y_mat, alpha, box_constraints)
                J_cross *= -1
                print ('    new J cross', J_cross)

            #Line search along D for gamma (step) in [0, gamma_max]
            # gamma = helpers.get_armijos_step_size()
            gamma = helpers.get_armijos_step_size(kernel_matrices, d, y_mat, alpha,
                                                  box_constraints, gamma_max, J_cross,
                                                  D, dJ)
            print ('gamma:', gamma)
            print ('D:', D)
            d += gamma * D
            d = helpers.fix_precision_of_vector(d, 1)

        print(d)
        self.weights = d
        ker_matrix = summation(self.KL, self.weights)
        self.ker_matrix = ker_matrix
        #Return final weights
        return ker_matrix

        Y = [1 if y==self.classes_[1] else -1 for y in self.Y]
        n_sample = len(self.Y)
        ker_matrix = matrix(summation(self.KL))
        YY = spdiag(Y)
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = spdiag([self.lam]*n_sample)
        Q = 2*(KLL+LID)
        p = matrix([0.0]*n_sample)
        G = -spdiag([1.0]*n_sample)
        h = matrix([0.0]*n_sample,(n_sample,1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in Y],[1.0 if lab2==-1 else 0 for lab2 in Y]]).T
        b = matrix([[1.0],[1.0]],(2,1))
         
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = self.max_iter
        sol = solvers.qp(Q,p,G,h,A,b)
        gamma = sol['x']
        if self.verbose:
            print ('[EasyMKL]')
            print ('optimization finished, #iter = %d' % sol['iterations'])
            print ('status of the solution: %s' % sol['status'])
            print ('objval: %.5f' % sol['primal objective'])

        yg = gamma.T * YY
        weights = [(yg*matrix(K.astype(np.double))*yg.T)[0] for K in self.KL]
         
        norm2 = sum([w for w in weights])
        self.weights = np.array([w / norm2 for w in weights])
        ker_matrix = summation(self.KL, self.weights)
        self.ker_matrix = ker_matrix
        return ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam,
                "generator": self.generator, "max_iter":self.max_iter,
                "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
                'estimator':self.estimator}
