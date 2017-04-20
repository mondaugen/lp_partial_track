import numpy as np
import itertools as it
import cvxopt
import matplotlib.pyplot as plt
import lpmisc
import string
from numpy import linalg
import sigmod as sm
import ddm

def get_lp_mats(i_frames,i_vals,dfunc,dmax,L):

    """
    i_frames:
        indices in each frame
    i_vals:
        values for each index such that the value of an index can be looked up
        as i_vals[idx]
    dfunc:
        function that accepts two entries from i_vals and gives distance between
        them
    dmax:
        the maximum tolerated distance
    L:
        the number of paths to find

    returns:
        (c,G,h,A,b,M,i_pairs,i_costs)

        c:
            cost vector
        G: 
            inequality constraint matrix
        h:
            inequality constraint vector
        A:
            equality constraint matrix
        b:
            equality constraint vector
        M:
            number of pairs considered
        i_pairs:
            the pairs of indices
        i_costs:
            the cost of each connection between the respective pairs

        Works with both simplex and interior-point methods.
    """

    # number of frames
    F=len(i_frames)

    # compute index pairs
    i_pairs=list(it.imap(lambda x,y : list(it.product(x,y)), i_frames[:-1], i_frames[1:]))
    i_pairs=reduce(lambda x,y : x+y, i_pairs)
    fv_dict=dict()
    i_all=reduce(lambda x,y : x+y, i_frames)
    for idx,val in zip(i_all,i_vals):
        fv_dict[idx]=val

    # compute costs for each pair
    i_costs=[float("inf") for _ in i_pairs]
    for idx,pair in enumerate(i_pairs):
        i,j=pair
        i_costs[idx]=dfunc(i_vals[i],i_vals[j])

    # filter excessive costs
    fltr_v=[i_cost < dmax for i_cost in i_costs]
    i_pairs=list(it.compress(i_pairs,fltr_v))
    i_costs=list(it.compress(i_costs,fltr_v))

    # number of pairs
    M=len(i_pairs)

    # i_costs is cost vector of LP
    c=np.array(i_costs)

    # number of incoming connections constraints
    r_p=sum([len(i_f) for i_f in i_frames[1:]])
    Ap=np.ndarray((r_p,len(i_pairs)))
    for r, i_p in enumerate(reduce(lambda x,y : x+y, i_frames[1:])):
        Ap[r,:]=[1. if x[1]==i_p else 0. for x in i_pairs]
    bp=np.ones((r_p,1))

    # number of outgoing connections constraints
    r_s=sum([len(i_f) for i_f in i_frames[:-1]])
    As=np.ndarray((r_s,len(i_pairs)))
    for r, i_s in enumerate(reduce(lambda x,y : x+y, i_frames[:-1] )):
        As[r,:]=[1. if x[0]==i_s else 0. for x in i_pairs]
    bs=np.ones((r_s,1))

    # balanced number of incoming and outgoing connections for inner frames
    r_b=sum([len(i_f) for i_f in i_frames[1:-1]])
    Ab=Ap[:-len(i_frames[-1]),:]-As[len(i_frames[0]):,:]
    bb=np.zeros((r_b,1))

    # total number of connections for each frame
    Ac=np.ndarray((F-1,len(i_pairs)))
    i_acc=0
    for f, f_i in enumerate(i_frames[:-1]):
        Ac[f,:] = np.sum(As[i_acc:i_acc+len(f_i),:],0)
        i_acc += len(f_i)
    bc=np.ones((F-1,1))*L

    # Also need 0 <= x <= 1
    I=np.eye(M)
    v1=np.ones((M,1))

    # build inequality contraint matrix
    G=np.vstack((As,Ap,-I))
    # and its vector
    h=np.vstack((bs,bp,0*v1))

    # Build equality contraint matrix
    A=np.vstack((Ab,Ac[-1,:]))
    b=np.vstack((bb,bc[-1,:]))

    return (c,G,h,A,b,M,i_pairs,i_costs)

def get_mq_sol(i_frames,i_vals,dfunc,dmax,L):
    """
    i_frames:
        indices in each frame. Must be contiguous and start at 0, e.g.,
        [[0,1],[2,3,4],...]
    i_vals:
        values for each index such that the value of an index can be looked up
        as i_vals[idx]
    dfunc:
        function that accepts two entries from i_vals and gives distance between
        them
    dmax:
        the maximum tolerated distance
    L:
        the number of paths to find

    returns (x,D)

    x:
        a list of lists of indices representing L best paths sorted from
        shortest to longest
    D:
        the cost tensor for all the paths
    """
    dim_sizes=[len(d) for d in i_frames]
    for dim_idcs,fr_idcs in zip(it.product(*map(xrange,dim_sizes)),it.product(*i_frames)):
        vals_pairs=[(i_vals[i],
            i_vals[j]) for i,j in zip(fr_idcs[:-1],fr_idcs[1:])]
        D.itemset(dim_idcs,sum([dfunc(iv,jv) for iv,jv in vals_pairs]))
    x=[]
    all_dim_idcs=list(it.product(*map(xrange,dim_sizes)))
    for j in xrange(L):
        min_cost=float('inf')
        min_idcs=None
        for dim_idcs in all_dim_idcs:
            if (D.item(dim_idcs) < min_cost):
                min_cost = D.item(dim_idcs)
                min_idcs = dim_idcs
        if min_cost > dmax:
            break
        x.append(min_idcs)
        all_dim_idcs.remove(min_idcs)
    return x

def estimate_ddm_decomp(x,
                        Fs=16000,
                        M=1024,
                        H=256,
                        wname='c1-nuttall-4',
                        b_ddm_hz=150.,
                        o_ddm_hz=75.,
                        th_ddm=10.**(-20./20)):
    """
    Decompose signal into frequency and amplitude modulated components.

    x:
        The signal to analyse.
    Fs:
        The sample rate.
    M:
        The analysis window size.
    H:
        The hop size.
    wname:
        The name of the analysis window to use 
        (see ddm.w_dw_sum_cos)
    b_ddm_hz:
        Size of band over which local maximum is searched (in Hz)
    o_ddm_hz:
        Spacing between the start points of these bands (in Hz)
    th_ddm:
        Threshold of value seen as valid.
    """
    ## Find maxima and estimate parameters
    # compute windows
    w,dw=ddm.w_dw_sum_cos(M,'c1-nuttall-4')#'hanning')
    # Convert to bins
    b_ddm=np.round(b_ddm_hz/Fs*M)
    o_ddm=np.round(o_ddm_hz/Fs*M)
    # Highest bin to consider
    M_ddm=M/2
    # number of bins after last maximum to skip
    i_ddm=3
    a=[]
#    a0=[]
    # length of signal
    N=len(x)
    # current hop
    h=0
    while ((h+M) <= N):
        #a0.append(
        a0=sm.ddm_p2_1_3_b(x[h:(h+M)],w,dw,
            b_ddm,o_ddm,th_ddm,M_ddm,i_ddm,norm=True)
        #)
        h+=H
        #a.append(a0[-1])
        a.append(a0)
    return a

