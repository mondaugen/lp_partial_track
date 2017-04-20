# Try doubling up some points in the middle to see if this makes enough
# room for all the paths
import matplotlib.pyplot as plt
import numpy as np
import bestpathslp as bplp
import cvxopt

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
        o_ddm_hz=75.,
        Fs=Fs)

# Number of paths L
L=6

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

ndp=sum([len(x) for x in a_flt])
print 'number of data points %d' % (ndp,)
print 'average number per frame %d' % (ndp/len(a_flt),)
print 'number of frames %d' % (len(a_flt),)

fig,ax1=plt.subplots(1,1)
h=0
for a_ in a_flt:
    for a__ in a_:
        f0_=np.imag(a__[1])/(2.*np.pi)*Fs
        f1_=(np.imag(a__[1])+2.*np.imag(a__[2])*H)/(2.*np.pi)*Fs
        t0_=h/H#float(Fs)
        t1_=(h+H)/H#float(Fs)
        ax1.plot([t0_,t1_],[f0_,f1_],c='k')
    h+=H

# Do partial tracking
# reduce lists into one big list
i_vals=reduce(lambda x,y : x+y,a_flt)
# produce indices for each frame
# start with all indices
i_frames_=range(len(i_vals))
i_frames=[]
# times at which corresponding points happen
i_times=[]
h=0

# frames in which to double the points
i_double=range(20,31)

for i in i_double:
    atlen=len(a_flt[i])
    for j in xrange(atlen):
        a_flt[i].append(a_flt[i][j].copy())

for a_ in a_flt:
    i_frames.append(i_frames_[:len(a_)])
    i_times += [h for _ in xrange(len(a_))]
    h+=H
    del i_frames_[:len(a_)]

def dfunc(a,b):
    return (abs((np.imag(a[1])+2.*np.imag(a[2])*H)
            -np.imag(b[1])) + abs(np.imag(a[2]) 
                - np.imag(b[2])))

(c,G,h,A,b,M,i_pairs,i_costs)=bplp.get_lp_mats(i_frames,
                                               i_vals,
                                               dfunc,
                                               0.1,
                                               L)
print 'max cost %f' % (max(i_costs),)
print 'min cost %f' % (min(i_costs),)

G_=cvxopt.sparse(cvxopt.matrix(G))
h_=cvxopt.matrix(h)
A_=cvxopt.sparse(cvxopt.matrix(A))
b_=cvxopt.matrix(b)
c_=cvxopt.matrix(c+1)

sol=cvxopt.solvers.lp(c_,G_,h_,A=A_,b=b_)

for x_pt in zip(sol['x'],i_pairs):
    x,pt = x_pt
    i,j=pt
    if x > 0.5:
        f0_=np.imag(i_vals[i][1])/(2.*np.pi)*Fs
        f1_=np.imag(i_vals[j][1])/(2.*np.pi)*Fs
        t0_=i_times[i]/H#float(Fs)
        t1_=i_times[j]/H#float(Fs)
        ax1.plot([t0_,t1_],[f0_,f1_],c='r')

if (show_plot):
    plt.show()
