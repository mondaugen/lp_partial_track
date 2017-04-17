# Miscellaneous LP stuff
import cvxopt
import numpy as np

def check_if_redundant(A,b,s,t):
    """
    Given system of linear inequalities A*x <= b determine if s^T*x <= t is a
    redundant inequality. Returns True if redundant, else False.

    A,b,s must be of type numpy.ndarray and t must be a float.
    s must be row vector

    """
    G=np.vstack((A,s))
    h=np.vstack((b,np.array([[t+1]])))
    s_=cvxopt.matrix(s.reshape((s.shape[1],1)))
    G_=cvxopt.sparse(cvxopt.matrix(G))
    h_=cvxopt.matrix(h)
    sol=cvxopt.solvers.lp(-1*s_,G_,h_,solver='glpk')
    #print 'primal objective %f' % (sol['primal objective'],)
    #print 'sol[x]: '
    #print sol['x']
    # primal objective negated because lp solver can only minimize
    return -sol['primal objective'] <= t

def check_if_redundant_test():
    A=np.array([[1,1],[-1,1]])
    # this is redundant
    s=np.array([[1./3,1/2.]])
    b=np.array([1,1]).reshape((2,1))
    t=1.
    if check_if_redundant(A,b,s,t):
        print 'redundant'
    else:
        print 'not redundant'
    A=np.array([[1,1],[-1,1]])
    # this is not redundant
    s=np.array([[1./3,2.]])
    b=np.array([1,1]).reshape((2,1))
    t=1.
    if check_if_redundant(A,b,s,t):
        print 'redundant'
    else:
        print 'not redundant'
