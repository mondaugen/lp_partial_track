import numpy as np
import itertools as it
import cvxopt
import matplotlib.pyplot as plt
import lpmisc
import string
from numpy import linalg

def print_0_1_array(x,remove_0s=False):
    """
    Prints an array containing only 0s and 1s in a consise way.
    """
    for row in x:
        s=""
        for col in row:
            s += "%2.0f" % (col,)
        if (remove_0s):
            s=s.translate(string.maketrans("0"," "))
        print s

def get_L_best_paths_mats(i_frames,i_vals,dfunc,dmax,L):

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

        Works with both simlpex and interior-point methods.
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
#    G=np.vstack((As,Ap,I,-I))
    G=np.vstack((As,Ap,-I))
    # and its vector
#    h=np.vstack((bs,bp,v1,0*v1))
    h=np.vstack((bs,bp,0*v1))
    # Build equality contraint matrix
    A=np.vstack((Ab,Ac[-1,:]))
    b=np.vstack((bb,bc[-1,:]))
    return (c,G,h,A,b,M,i_pairs,i_costs)

N_frames = 4
N_pts_per_frame = 7
x_pts=np.add.outer(np.arange(N_frames),np.zeros(N_pts_per_frame)).flatten()
y_pts=np.multiply.outer(np.ones((N_frames,1)),N_pts_per_frame-1-np.arange(N_pts_per_frame)).flatten()
#y_pts+=0.1*np.random.standard_normal(y_pts.size)
#y_pts=np.random.uniform(0,N_pts_per_frame,N_frames*N_pts_per_frame)
i_vals=zip(x_pts,y_pts)
def dfun (p1,p2): return np.sqrt((p1[0]-p2[0])**2. + (p1[1]-p2[1])**2.)
#dmax=100.
dmax=np.sqrt(2.)+0.01
L=3
n_pts_per_frame=[N_pts_per_frame for _ in xrange(N_frames)]
i_frames = [range(l,k) for l,k in
        zip(np.cumsum([0]+n_pts_per_frame)[:-1],np.cumsum(n_pts_per_frame))]
(c,G,h,A,b,M,i_pairs,i_costs)=get_L_best_paths_mats(i_frames,i_vals,dfun,dmax,L)

## determine which rows redundant

find_redund=False
if (find_redund):
    i_red=[]
    i_row=0
    i_ghost_row=0 # the row in the original matrix
    G_o=G.copy()
    while True:
        row=G[i_row,:]
        t_t=h[i_row,0]
        G_r=np.vstack((G[:i_row,:],G[i_row+1:,:]))
        h_r=np.vstack((h[:i_row,:],h[i_row+1:,:]))
        if (lpmisc.check_if_redundant(G_r,h_r,row.reshape((1,row.shape[0])),t_t)):
            i_red.append(i_ghost_row)
            G=G_r
            h=h_r
        else:
            i_row += 1
        i_ghost_row += 1
        if i_row >= G.shape[0]:
            break

    print "redundant:"
    print i_red

H_phi=np.dot(G.T,G)
print_0_1_array(np.linalg.cholesky(H_phi),remove_0s=False)

G_=cvxopt.sparse(cvxopt.matrix(G))
h_=cvxopt.matrix(h)
A_=cvxopt.sparse(cvxopt.matrix(A))
b_=cvxopt.matrix(b)
c_=cvxopt.matrix(c+1)
opt_solver=None#'glpk'
sol=cvxopt.solvers.lp(c_,G_,h_,A=A_,b=b_,solver=opt_solver)
plt.clf()
plt.plot(x_pts,y_pts,'bo')
print '# iterations = %d' % sol['iterations']
for k,x_pt in enumerate(zip(sol['x'],i_pairs)):
    x,pt = x_pt
    i,j=pt
    if x > 0.5:
        plt.plot(x_pts[[i,j]],y_pts[[i,j]],'k')
    else:
        plt.plot(x_pts[[i,j]],y_pts[[i,j]],'LightGray')
plt.show()
