import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from scipy import linalg

def print_0_1_array(x):
    """
    Prints an array containing only 0s and 1s in a consise way.
    """
    for row in x:
        s=""
        for col in row:
            s += "%2.0f" % (col,)
        print s

def get_L_best_paths_mats(c,n_pts_per_frame,L):
    """
    Get the matrices for the linear program that finds the L best paths through
    a lattice.

    c:
        The cost tensor flattened into an array. Let C denote this tensor which
        must have size MxM. If C[i,j] is the cost between node i and node j,
        then c[M*i + j] = C[i,j].
    n_pts_per_frame:
        A list of the number of nodes in each frame. The numbers correspond to
        the number of contiguous indices of the axes in C, e.g., if
        n_pts_per_frame is [2,4,3] then frame 0 contains nodes [0,1], frame 1
        [2,3,4,5], and frame 2 [6,7,8].
    L:
        The number of paths.

    returns tuple (c,G,h,A,b)

    c:
        The cost vector augmented to extract a path.
    G:
        The inequality constraint matrix such that G*x <= h.
    h:
        The inequality constraint vector.
    A:
        The equality constraint matrix such that A*x = b.
    b:
        The equality constraint vector.
    """
    # total number of nodes
    M=sum(n_pts_per_frame)
    # number of frames
    F=len(n_pts_per_frame)
    if not isinstance(c,np.ndarray):
        raise Exception("c must be type np.ndarray, currently is type %s." %
                (type(c),))
    if (F < 3):
        raise Exception("Must use 3 or more frames, here F=%d.", (F,))
    if (c.size != (M*M)):
        raise Exception("Bad size %d for c or total sum %d for n_pts_per_frame." %
                (c.size,M))
    if (L > min(n_pts_per_frame)):
        raise Exception(
        "Number of points in frame %d less than number of paths L=%d."
        "No solution possible." % (np.argmin(n_pts_per_frame),L))
            
    # point indices in each frame
    pts_per_frame = [range(l,k) for l,k in
            zip(np.cumsum([0]+n_pts_per_frame)[:-1],np.cumsum(n_pts_per_frame))]
    # incoming connections
    # matrix (first frame can't have incoming cxns)
    Ap=np.zeros((sum(n_pts_per_frame[1:]),M*M))
    r=0
    for i in xrange(len(pts_per_frame)-1):
        fpts=pts_per_frame[i+1]
        for fpt in fpts:
            Ap[r,M*np.array(pts_per_frame[i])+fpt]=1
#            Ap[r,M*np.array(range(M))+fpt]=1
            r+=1
    # vector
    bp=np.ones((np.sum(n_pts_per_frame[1:]),1))
    # outgoing connections
    # matrix (last frame can't have outgoing cxns)
    As=np.zeros((sum(n_pts_per_frame[:-1]),M*M))
    r=0
    for i in xrange(len(pts_per_frame)-1):
        fpts=pts_per_frame[i]
        for fpt in fpts:
            As[r,M*fpt + np.array(pts_per_frame[i+1])]=1
#            As[r,M*fpt + np.array(range(M))]=1
            r+=1
    # matrix (first frame can't have incoming cxns)
    Ap_=np.zeros((sum(n_pts_per_frame[1:]),M*M))
    r=0
    for i in xrange(len(pts_per_frame)-1):
        fpts=pts_per_frame[i+1]
        for fpt in fpts:
#            Ap_[r,M*np.array(pts_per_frame[i])+fpt]=1
            _ins=np.array(list(set(range(M))-set([fpt])))
            Ap_[r,M*_ins+fpt]=1
            r+=1
    # vector
    bp=np.ones((np.sum(n_pts_per_frame[1:]),1))
    # outgoing connections
    # matrix (last frame can't have outgoing cxns)
    As_=np.zeros((sum(n_pts_per_frame[:-1]),M*M))
    r=0
    for i in xrange(len(pts_per_frame)-1):
        fpts=pts_per_frame[i]
        for fpt in fpts:
#            As_[r,M*fpt + np.array(pts_per_frame[i+1])]=1
            _ins=np.array(list(set(range(M))-set([fpt])))
            As_[r,M*fpt + _ins]=1
            r+=1
    # vector
    bs=np.ones((np.sum(n_pts_per_frame[:-1]),1))
    # equal number of incoming and outgoing connections for inner frames
    # matrix
    Ab=Ap[:-n_pts_per_frame[-1],:]-As[n_pts_per_frame[0]:,:]
    Ab_=Ap_[:-n_pts_per_frame[-1],:]-As_[n_pts_per_frame[0]:,:]
    # vector
    bb=np.zeros((sum(n_pts_per_frame[1:-1]),1))
    # constrain number of non-zero values in each frame to be L (for L paths)
    # matrix
    Ac=np.zeros((F-1,M*M))
    Ac_=np.zeros((F-1,M*M))
    # count by aribtrarily couning the number of outgoing connections, except
    # for the last frame
    for f,fpts in enumerate(pts_per_frame[:-1]):
        Ac_[f,:]=np.sum(As_[fpts,:],axis=0) # sum along rows
    for f,fpts in enumerate(pts_per_frame[:-1]):
        Ac[f,:]=np.sum(As[fpts,:],axis=0) # sum along rows
    # vector
    bc=L*np.ones((F-1,1))

    # Also need 0 <= x <= 1
    I=np.eye(M*M)
    v1=np.ones((M*M,1))
    # build inequality contraint matrix
    G=np.vstack((As,Ap,-As,-Ap,I,-I))
    G=linalg.block_diag(G,G)
    G=np.hstack((G,np.zeros((G.shape[0],M*M))))
    G=np.vstack((G,np.hstack((I,-I,-I))))
    G=np.vstack((G,np.hstack((-I,I,-I))))
    G=np.vstack((G,np.hstack((0*I,0*I,-I))))
    # Constrain error to be less than path_lengths * (num_paths -
    # num_extracted_paths). Should be feasible.
    G=np.vstack((G,np.zeros((1,3*M*M))))
    G[-1,2*M*M:3*M*M]=1.
    G[-1,-1]=-(F-1)
    G=np.hstack((G,np.zeros((G.shape[0],1))))
    # and its vector
    h=np.vstack((bs,bp,0*bs,0*bp,v1,0*v1))
    h=np.vstack((h,h,np.zeros((3*M*M,1))))
    # error constraint
    h=np.vstack((h,np.array([[(F-1)*(L-1)]])))
    # Build equality contraint matrix
    A=np.vstack((Ab_,Ac))
    A=linalg.block_diag(A,np.vstack((Ab_,Ac)))
    A=np.hstack((A,np.zeros((A.shape[0],M*M))))
    A=np.hstack((A,np.zeros((A.shape[0],1))))
    A[-Ac.shape[0],-1]=-1
    _z=np.zeros((3,3*M*M+1))
    pth_idx=0
    # get whether or not path starts on node 0
    _z[0,1:M]=1
    _z[0,-1]=-1
    # if so, y must have a connection from node 0
    _z[1,M*M:(M*M+M)]=1
    _z[1,-1]=-1
#    # and y can contain a maximum of F-2 ones
    _z[2,M*M+M*n_pts_per_frame[0]:2*M*M]=1
    _z[2,-1]=-(F-2)
#    _z[1,M*M:2*M*M]=1
#    _z[1,-1]=-(F-1)
#    _z[0,M*pth_idx:(pth_idx*M+M)]=1
    #_z[0,:M]=-1
    #_z[0,M*M:(M*M+M)]=1
    #_z[1,M*M+M:M*M+M*n_pts_per_frame[0]]=1
    A=np.vstack((A,_z))
    b=np.vstack((bb,bc))
    b=np.vstack((b,bb,np.zeros(bc.shape)))
#    b=np.vstack((b,np.array([[0]])))
    b=np.vstack((b,np.array([[0],[0],[0]])))
#    b=np.vstack((b,bb,np.zeros(bc.shape),np.array([[0],[0]])))
    c_=np.zeros((3*M*M+1,1))
    c_[:M*M,0]=c
    c_[-(M*M+1):-1,0]=1
#    c_[-1]=-1*sum(c)
#    c_[M*M:2*M*M]=-1
    return (c_,G,h,A,b,M)

N_frames = 5
N_pts_per_frame = 6
x_pts=np.add.outer(np.arange(N_frames),np.zeros(N_pts_per_frame)).flatten()
y_pts=np.random.uniform(-1,1,N_frames*N_pts_per_frame)
# Make first point really far out
#y_pts[0]=10.
c=np.power(np.subtract.outer(x_pts,x_pts),2.)+np.power(np.subtract.outer(y_pts,y_pts),2.)
c=c.flatten()
n_pts_per_frame=[N_pts_per_frame for _ in xrange(N_frames)]
L=5
(c,G,h,A,b,M)=get_L_best_paths_mats(c,n_pts_per_frame,L)
G_=cvxopt.sparse(cvxopt.matrix(G))
h_=cvxopt.matrix(h)
A_=cvxopt.sparse(cvxopt.matrix(A))
b_=cvxopt.matrix(b)
c_=cvxopt.matrix(c+1)
opt_solver=None#'glpk'
sol=cvxopt.solvers.lp(c_,G_,h_,A=A_,b=b_,solver=opt_solver)
plt.clf()
plt.plot(x_pts,y_pts,'bo')
for label, pts in enumerate(zip(x_pts,y_pts)):
    x,y=pts
    plt.annotate(
        "%d" % (label,),
        xy=(x, y), xytext=(-10, 10),
        textcoords='offset points', ha='right', va='bottom',
       # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
       # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
        )
for k,x in enumerate(sol['x'][:M*M]):
    if x > 0.5:
        i = k / M
        j = k % M
        plt.plot(x_pts[[i,j]],y_pts[[i,j]],'k')
for k,x in enumerate(sol['x'][M*M:2*M*M]):
    if x > 0.5:
        i = k / M
        j = k % M
        plt.plot(x_pts[[i,j]],y_pts[[i,j]],'r')


plt.show()
