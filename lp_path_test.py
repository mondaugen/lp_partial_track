import numpy as np
import matplotlib.pyplot as plt
import cvxopt

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

    returns tuple (G,h,A,b)

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
            r+=1
    # vector
    bs=np.ones((np.sum(n_pts_per_frame[:-1]),1))
    # equal number of incoming and outgoing connections for inner frames
    # matrix
    Ab=Ap[:-n_pts_per_frame[-1],:]-As[n_pts_per_frame[0]:,:]
    # vector
    bb=np.zeros((sum(n_pts_per_frame[1:-1]),1))
    # constrain number of non-zero values in each frame to be L (for L paths)
    # matrix
    Ac=np.zeros((F,M*M))
    # count by aribtrarily couning the number of outgoing connections, except
    # for the last frame
    for f,fpts in enumerate(pts_per_frame[:-1]):
        Ac[f,:]=np.sum(As[fpts,:],axis=0) # sum along rows
    # in last frame there are only incoming connections so use these to count
    Ac[-1,:]=np.sum(Ap[-n_pts_per_frame[-1],:],axis=0)
    # vector
    bc=L*np.ones((F,1))
    # Number of outgoing connections to non-contiguous frames constrained to 0
    Anoc=np.zeros((sum(n_pts_per_frame[:-2]),M*M))
    r=0
    for i in xrange(len(pts_per_frame)-2):
        fpts=pts_per_frame[i]
        if (i == (len(pts_per_frame)-3)):
            rest_pts=pts_per_frame[-1]
        else:
            rest_pts=reduce(lambda x,y : x+y, pts_per_frame[i+2:]) # note this performs list concatenation
        for fpt in fpts:
            Anoc[r,M*fpt + np.array(rest_pts)]=1
            r+=1
    bnoc=np.zeros((Anoc.shape[0],1))
    # Number of incoming connections to non-contiguous frames constrained to 0
    Anic=np.zeros((sum(n_pts_per_frame[2:]),M*M))
    r=0
    for i in xrange(len(pts_per_frame)-2):
        fpts=pts_per_frame[i+2]
        if (i == 0):
            rest_pts=pts_per_frame[0]
        else:
            rest_pts=reduce(lambda x,y : x+y, pts_per_frame[i+2:])
        for fpt in fpts:
            Anic[r,M*np.array(rest_pts) + fpt]=1
            r+=1
    bnic=np.zeros((Anic.shape[0],1))

    # Also need 0 <= x <= 1
    I=np.eye(M*M)
    v1=np.ones((M*M,1))
    # build inequality contraint matrix
    G=np.vstack((As,Ap,-As,-Ap,I,-I))
    # and its vector
    h=np.vstack((bs,bp,0*bs,0*bp,v1,0*v1))
    # Build equality contraint matrix
    A=np.vstack((Ab,Ac,Anoc,Anic))
    b=np.vstack((bb,bc,bnoc,bnic))
    return (G,h,A,b)

N_frames = 8
N_pts_per_frame = 5
x_pts=np.add.outer(np.arange(N_frames),np.zeros(N_pts_per_frame)).flatten()
y_pts=np.random.uniform(0,1,N_frames*N_pts_per_frame)
c=np.power(np.subtract.outer(x_pts,x_pts),2.)+np.power(np.subtract.outer(y_pts,y_pts),2.)
c=c.flatten()
#plt.plot(x_pts,y_pts,'bo')
#plt.show()
n_pts_per_frame=[N_pts_per_frame for _ in xrange(N_frames)]
L=3
(G,h,A,b)=get_L_best_paths_mats(c,n_pts_per_frame,L)
G=cvxopt.matrix(G)
h=cvxopt.matrix(h)
A=cvxopt.matrix(A)
b=cvxopt.matrix(b)
c=cvxopt.matrix(c)
sol=cvxopt.solvers.lp(c,G,h,A=A,b=b)
