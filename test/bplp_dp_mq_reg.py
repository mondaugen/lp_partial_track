# Try doubling up some points in the middle to see if this makes enough
# room for all the paths
# compare with MQ algorithm
# Regularize cost functions
import matplotlib.pyplot as plt
import numpy as np
import bestpathslp as bplp
import cvxopt
import sys

show_plot=True

#fname='/Users/mondaugen/Documents/sound/timit/TIMIT/TEST/DR1/FAKS0/SA1_your_dark.f64'
fname='sounds/chirps.f64'
x=np.fromfile(fname,dtype='float64')
sigpow=np.sum(x*x)/len(x)
SNR=-0
noisepow=sigpow*np.power(10.,SNR/10.)
# add noise
x+=np.random.standard_normal(len(x))*np.sqrt(noisepow)
x=x[:len(x)]
Fs=16000
H=512
M=2048
a=bplp.estimate_ddm_decomp(x,
        M=M,
        H=H,
        th_ddm=10.**(-100./20),
        b_ddm_hz=100.,
        o_ddm_hz=50.,
        Fs=Fs)

# Number of paths L
L=6

# make colours
cmap=plt.get_cmap('magma')

# Boundaries on frequencies to consider
f_min=250.
f_max=2250
# Keep only data-points within bounds
def _dp_filt(_a):
    f0_=np.imag(_a[1])/(2.*np.pi)*Fs
    f1_=(np.imag(_a[1])+2.*np.imag(_a[2])*H)/(2.*np.pi)*Fs
    return ((max(f0_,f1_) < f_max)
        and (min(f0_,f1_) > f_min))
a_flt=[]
for a_ in a:
    a_flt.append(filter(_dp_filt,a_))

fig,(ax1,ax2)=plt.subplots(1,2)
h=0
for a_ in a_flt:
    for a__ in a_:
        f0_=np.imag(a__[1])/(2.*np.pi)*Fs
        f1_=(np.imag(a__[1])+2.*np.imag(a__[2])*H)/(2.*np.pi)*Fs
        t0_=h/H#float(Fs)
        t1_=(h+H)/H#float(Fs)
        ax1.plot([t0_,t1_],[f0_,f1_],c='LightGrey')
        ax2.plot([t0_,t1_],[f0_,f1_],c='LightGrey')
    h+=H

# Do partial tracking (LP)
# reduce lists into one big list
i_vals=reduce(lambda x,y : x+y,a_flt)
# produce indices for each frame
# start with all indices
i_frames_=range(len(i_vals))
i_frames=[]
# times at which corresponding points happen
i_times=[]
h=0

for a_ in a_flt:
    i_frames.append(i_frames_[:len(a_)])
    i_times += [h for _ in xrange(len(a_))]
    h+=H
    del i_frames_[:len(a_)]

def dfunc(a,b):
#    f0=np.imag(a[1])/(2.*np.pi)*Fs
#    f1_pr=f0+(np.imag(a[1])+2.*np.imag(a[2])*H)/(2.*np.pi)*Fs
#    f1=np.imag(b[1])/(2.*np.pi)*Fs
#    return abs(f1_pr - f1) + np.random.uniform()*1e-3
    return (abs((np.imag(a[1])+2.*np.imag(a[2])*H)
            -np.imag(b[1])) + abs(np.imag(a[2]) 
                - np.imag(b[2]))*10. + np.random.uniform()*1e-5)

# do MQ algorithm
# only do sets of K frames or else algorithm intractable

K=2
for k in xrange(0,len(i_frames),K-1):
    (sl,D_)=bplp.get_mq_sol(i_frames[k:k+K],
                            i_vals,
                            dfunc,
                            float('inf'),
                            L)
    pths_=[[(i,j) for i,j in zip(sl_[:-1],sl_[1:])] for sl_ in sl]
    for i,pth in enumerate(pths_):
        clr=cmap(float(i+1)/(L+1))
        for pr in pth:
            j,k=pr
            f0_=np.imag(i_vals[j][1])/(2.*np.pi)*Fs
            f1_=np.imag(i_vals[k][1])/(2.*np.pi)*Fs
            t0_=i_times[j]/H#float(Fs)
            t1_=i_times[k]/H#float(Fs)
            ax2.plot([t0_,t1_],[f0_,f1_],c=clr,lw=2.)

# Do partial tracking (LP)
# reduce lists into one big list
i_vals=reduce(lambda x,y : x+y,a_flt)
# produce indices for each frame
# start with all indices
i_frames_=range(len(i_vals))
i_frames=[]
# times at which corresponding points happen
i_times=[]
h=0

for a_ in a_flt:
    i_frames.append(i_frames_[:len(a_)])
    i_times += [h for _ in xrange(len(a_))]
    h+=H
    del i_frames_[:len(a_)]

# frames in which to double the points
i_double=range(22,34)

# double the points to have overlapping paths
for i in i_double:
    atlen=len(a_flt[i])
    for j in xrange(atlen):
        a_flt[i].append(a_flt[i][j].copy())

ndp=sum([len(x) for x in a_flt])
print 'number of data points %d' % (ndp,)
print 'average number per frame %d' % (ndp/len(a_flt),)
print 'number of frames %d' % (len(a_flt),)

# do a couple times just to get costs
def dfunc_pr(a,b):
    """ Prediction error """
    return abs((np.imag(a[1])+2.*np.imag(a[2])*H)-np.imag(b[1]))

def dfunc_slp(a,b):
    """ Slope matching """
    return abs(np.imag(a[2]) - np.imag(b[2]))

(c,G,h,A,b,M,i_pairs,i_costs_pr)=bplp.get_lp_mats(i_frames,
                                               i_vals,
                                               dfunc_pr,
                                               float('inf'),
                                               L,
                                               build_mats=False)
(c,G,h,A,b,M,i_pairs,i_costs_slp)=bplp.get_lp_mats(i_frames,
                                               i_vals,
                                               dfunc_slp,
                                               float('inf'),
                                               L,
                                               build_mats=False)
min_i_costs_pr=min(i_costs_pr)
max_i_costs_pr=max(i_costs_pr)
min_i_costs_slp=min(i_costs_slp)
max_i_costs_slp=max(i_costs_slp)
alph=0.75
def dfunc_reg(a,b):
    """ combination of prediction error and slope matching """
    return (alph * (dfunc_pr(a,b) - min_i_costs_pr)/(max_i_costs_pr-min_i_costs_pr)
    + (1. - alph) * (dfunc_slp(a,b) -
        min_i_costs_slp)/(max_i_costs_slp-min_i_costs_slp)
    + np.random.uniform()*1e-2)

(c,G,h,A,b,M,i_pairs,i_costs)=bplp.get_lp_mats(i_frames,
                                               i_vals,
                                               dfunc_reg,
                                               .23,
                                               L,
                                               build_mats=True)

print 'num costs: %d' % (len(i_costs),)
hi,be=np.histogram(i_costs,bins=10)
plt.figure(2)
plt.plot(be[:-1],hi)
plt.figure(1)

#plt.show()

print 'max cost %f' % (max(i_costs),)
print 'min cost %f' % (min(i_costs),)

G_=cvxopt.sparse(cvxopt.matrix(G))
h_=cvxopt.matrix(h)
A_=cvxopt.sparse(cvxopt.matrix(A))
b_=cvxopt.matrix(b)
c_=cvxopt.matrix(c+1)

sol=cvxopt.solvers.lp(c_,G_,h_,A=A_,b=b_)

(pths,i_pths)=bplp.extract_paths(sol['x'],i_pairs)

for i,pth,i_pth in zip(xrange(len(pths)),pths,i_pths):
    clr=cmap(float(i+1)/(L+1))
    for pr in pth:
        j,k=pr
        f0_=np.imag(i_vals[j][1])/(2.*np.pi)*Fs
        f1_=np.imag(i_vals[k][1])/(2.*np.pi)*Fs
        t0_=i_times[j]/H#float(Fs)
        t1_=i_times[k]/H#float(Fs)
        ax1.plot([t0_,t1_],[f0_,f1_],c=clr,lw=2.)





if (show_plot):
    plt.show()
