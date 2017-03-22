import ddm
import numpy as np
import matplotlib.pyplot as plt
import sigmod2 as sm
import sys
import ptpath
from cvxopt import solvers
import ptpath_test
import os
import neplot as nep
# Color contrast config
# values further from 1, more contrast
clr_gamma=4.
clr_mapper=nep.PowerNormalize(clr_gamma)

show_plot=True

plotoutpath='paper/plots/'
chirp_param_out_path=plotoutpath+'mq_lp_comp_smooth_chirp_params.txt'
if len(sys.argv) < 2:
    D_r=20.
    plotoutpath+='mq_lp_comp_smooth_chirp_'+str(int(np.round(D_r)))+'_dflt.eps'
else:
    D_r=float(sys.argv[1])
    plotoutpath+='mq_lp_comp_smooth_chirp_'+str(int(np.round(D_r)))+'.eps'

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex='col',sharey='row')

# Test McAulay and Quatieri partial path construction

# Synthesize signal
# Chirp 0
f0_0=500.
f0_1=600.
# Chirp 1
f1_0=700.
f1_1=400.
# Number of harmonics
K=3
# Boundaries on frequencies to consider
f_min=250.
f_max=2250

## Power of signal (dB)
#  Each sinusoid has amplitude 1 and there are K sinusoids per sound.
#  (The noise is added to each sound)
D_s=20.*np.log(K)
## Signal to noise ratio (dB)
D_n=D_s-D_r
P_n=10.**(D_n/10.)

T=1.
Fs=16000
N=int(np.round(T*Fs))
n=np.arange(N)

# Angular velocity as function of sample number, source 0
a0_1=f0_0/Fs*np.pi*2.
a0_2=(f0_1-f0_0)/Fs*np.pi*2./N
# Initial phase
a0_0=0.
# Angular velocity as function of sample number, source 1
a1_1=f1_0/Fs*np.pi*2.
a1_2=(f1_1-f1_0)/Fs*np.pi*2./N
# Initial phase
a1_0=0.
# Phase function
x0_n=np.zeros(N,dtype='complex_')
x1_n=np.zeros(N,dtype='complex_')
with open(chirp_param_out_path,'w') as fo:
    k_fo=0
    for k in np.arange(1,K+1):
        phi_n=np.polyval([0.5*a0_2*k,a0_1*k,a0_0],n)
        x0_n+=np.exp(1j*phi_n)
        #fo.write('%d & %2.2f & %2.2f & %2.2f $\\times 10^{-6}$ & %d & %d \\\\\n' %
        #        (k_fo,a0_0,a0_1*k,a0_2*k*1.e6,f0_0*k,f0_1*k))
        a0_2_exp=np.floor(np.log10(np.abs(a0_2*k)))
        fo.write('%d & %d & %2.2f & %2.2f $\\times 10^{%d}$ & %d & %d \\\\\n' %
                (k_fo,a0_0,a0_1*k,a0_2*k/(10.**a0_2_exp),int(a0_2_exp),f0_0*k,f0_1*k))
        k_fo+=1
    for k in np.arange(1,K+1):
        phi_n=np.polyval([0.5*a1_2*k,a1_1*k,a1_0],n)
        x1_n+=np.exp(1j*phi_n)
        #fo.write('%d & %2.2f & %2.2f & %2.2f $\\times 10^{-6}$ & %d & %d \\\\\n' %
        #        (k_fo,a1_0,a1_1*k,a1_2*k*1.e6,f1_0*k,f1_1*k))
        a1_2_exp=np.floor(np.log10(np.abs(a1_2*k)))
        fo.write('%d & %d & %2.2f & %2.2f $\\times 10^{%d}$ & %d & %d \\\\\n' %
                (k_fo,a1_0,a1_1*k,a1_2*k/(10.**a1_2_exp),int(a1_2_exp),f1_0*k,f1_1*k))
        k_fo+=1
#for k in np.arange(1,K+1):
#    phi_n=np.polyval([0.5*a0_2*k,a0_1*k,a0_0],n)
#    x0_n+=np.exp(1j*phi_n)
#for k in np.arange(1,K+1):
#    phi_n=np.polyval([0.5*a1_2*k,a1_1*k,a1_0],n)
#    x1_n+=np.exp(1j*phi_n)

# Add noise
x0_n+=np.random.standard_normal(N)*np.sqrt(P_n)
x1_n+=np.random.standard_normal(N)*np.sqrt(P_n)

# Analysis window length
M=1024
# Hop size
H=256

## Find maxima and estimate parameters
# compute windows
w,dw=ddm.w_dw_sum_cos(M,'c1-nuttall-4')#'hanning')
# Size of band over which local maximum is searched (in Hz)
b_ddm_hz=150.
# Spacing between the start points of these bands (in Hz)
o_ddm_hz=75.
# Convert to bins
b_ddm=np.round(b_ddm_hz/Fs*M)
o_ddm=np.round(o_ddm_hz/Fs*M)
print 'b_ddm, o_ddm = %d %d' %(b_ddm,o_ddm)
# threshold of value seen as valid
th_ddm=10.**(-20./20)
# Highest bin to consider
M_ddm=M/2
# number of bins after last maximum to skip
i_ddm=3

Pxx1, freqs1, frames1, im1 = ax1.specgram(x0_n+x1_n,NFFT=M,Fs=Fs,norm=clr_mapper,cmap='Greys')
ax2.specgram(x0_n+x1_n,NFFT=M,Fs=Fs,norm=clr_mapper,cmap='Greys')
ax3.specgram(x0_n+x1_n,NFFT=M,Fs=Fs,norm=clr_mapper,cmap='Greys')

a=[]
a0=[]
# current hop
h=0
while ((h+M) <= N):
    a0.append(sm.ddm_p2_1_3_b(x0_n[h:(h+M)],w,dw,
        b_ddm,o_ddm,th_ddm,M_ddm,i_ddm,norm=True))
    h+=H
    a.append(a0[-1])

a1=[]
h=0
# frame number
k=0
while ((h+M) <= N):
    a1.append(sm.ddm_p2_1_3_b(x1_n[h:(h+M)],w,dw,
        b_ddm,o_ddm,th_ddm,M_ddm,i_ddm,norm=True))
    h+=H
    a[k]+=a1[-1]
    k+=1

# Keep only data-points within bounds
def _dp_filt(_a):
    f0_=np.imag(_a[1])/(2.*np.pi)*Fs
    f1_=(np.imag(_a[1])+2.*np.imag(_a[2])*H)/(2.*np.pi)*Fs
    return ((max(f0_,f1_) < f_max)
        and (min(f0_,f1_) > f_min))

a_flt=[]
for a_ in a:
    a_flt.append(filter(_dp_filt,a_))

h=0
for a_ in a_flt:
    for a__ in a_:
        f0_=np.imag(a__[1])/(2.*np.pi)*Fs
        f1_=(np.imag(a__[1])+2.*np.imag(a__[2])*H)/(2.*np.pi)*Fs
        t0_=h/float(Fs)
        t1_=(h+H)/float(Fs)
        ax1.plot([t0_,t1_],[f0_,f1_],c='k')
    h+=H

# Find this many best paths
J=6
# Function for determining distance between 2 nodes
def _node_cxn_cost(a,b,dt):
#    return (((np.imag(a[1])+2.*np.imag(a[2])*0.5*H)
#             -(np.imag(b[1])+2.*np.imag(b[2])*0.5*H))**2.
#             + (2.*np.imag(a[2])
#             -2.*np.imag(b[2]))**2.)
#    return abs((np.imag(a[1])+2.*np.imag(a[2])*dt)
#            -np.imag(b[1]))#**2.
    return (abs((np.imag(a[1])+2.*np.imag(a[2])*dt)
            -np.imag(b[1])) + abs(np.imag(a[2]) - np.imag(b[2])))
# MQ method
A_cxn=[]
for k in xrange(len(a_flt)-1):
    m=len(a_flt[k])
    n=len(a_flt[k+1])
    A=np.ndarray((m,n),dtype=np.float64)
    for m_ in xrange(m):
        for n_ in xrange(n):
            A[m_,n_]=_node_cxn_cost(a_flt[k][m_],a_flt[k+1][n_],H)
    m_iter=list(xrange(m))
    n_iter=list(xrange(n))
    A_cxn_=[]
    for j in xrange(J):
        min_cost=float('inf')
        min_m=-1
        min_n=-1
        if (len(m_iter)>0) and (len(n_iter)>0):
            for m_ in m_iter:
                for n_ in n_iter:
                    if A[m_,n_] < min_cost:
                        min_m=m_
                        min_n=n_
                        min_cost=A[m_,n_]
            if (min_m>=0) and (min_n>=0):
                A_cxn_.append((min_m,min_n))
                m_iter.remove(min_m)
                n_iter.remove(min_n)
    A_cxn.append(A_cxn_)

h=0
for k,A_cxn_ in zip(xrange(len(a_flt)-1),A_cxn):
    for _cxn in A_cxn_:
        m_,n_=_cxn
        a_0=a_flt[k][m_]
        a_1=a_flt[k+1][n_]
        f0_=np.imag(a_0[1])/(2.*np.pi)*Fs
        f1_=np.imag(a_1[1])/(2.*np.pi)*Fs
        t0_=h/float(Fs)
        t1_=(h+H)/float(Fs)
        ax2.plot([t0_,t1_],[f0_,f1_],c='k')
    h+=H

## Compare with LP method
# Cost function
def _lp_cost_fun(a,b):
    return _node_cxn_cost(a.value,b.value,(b.frame_num-a.frame_num)*H)+1.
# Number of frames in LP
L=4
end_node_indices=[]
S_all=[]
F_all=[]
paths_all=[]
for l in xrange(0,len(a_flt)-L,L-1):
    a_flt_=a_flt[l:l+L]
    if (len(end_node_indices)<J):
        # There are not enough end_node_indices to account for all the desired
        # paths
        end_node_indices=list(xrange(len(a_flt_[0])))
    n_node=0
    n_frame=0
    # Build frames and graph
    F=[]
    S=dict()
    for a_ in a_flt_:
        F.append([])
        for k_a_ in xrange(len(a_)):
            if n_frame==0:
                # Only include node if it was an ending node of the last path
                if k_a_ not in end_node_indices:
                    continue
            a__=a_[k_a_]
            F[-1].append(n_node)
            S[n_node]=ptpath.LPNode(value=a__,frame_num=n_frame)
            n_node+=1
        n_frame+=1
    for l_ in xrange(L-1):
        for f in F[l_]:
            S[f].out_nodes=F[l_+1]
        for f in F[l_+1]:
            S[f].in_nodes=F[l_]
    # Build linear program
    d=ptpath.g_f_2lp(S,F,J,_lp_cost_fun,{'calc_mean':0,'min_mean_dev':0})
    # Solve LP
    sol=solvers.lp(d['c'],d['G'],d['h'],d['A'],d['b'])['x']
    # Extract paths
    paths=ptpath_test.lp_sol_extract_paths(sol,S,F)
    # Find path end nodes
    end_nodes=[]
    full_paths=[]
    for path in paths:
        if (len(path)==L):
            end_nodes.append(path[-1])
            full_paths.append(path)
    paths_all.append(full_paths)
    # Record indices in F[-1] of end_nodes
    end_node_indices=[F[-1].index(en) for en in end_nodes]
    S_all.append(S)
    F_all.append(F)

l=0
for S,F,paths in zip(S_all,F_all,paths_all):
    ts=np.array(list(xrange(l,l+L)),dtype=np.float64)*H/float(Fs)
    for path in paths:
        fs=[]
        for p in path:
            fs.append(np.imag(S[p].value[1])/(2.*np.pi)*Fs)
        fs=np.array(fs)
        ax3.plot(ts,fs,c='k')
        ax1.plot(ts,fs,c='r')
    l+=L-1

for l in xrange(0,len(a_flt)-L,L-1):
    ax3.plot([l*H/float(Fs),l*H/float(Fs)],[f_min,f_max],c='k',ls=':')

ax1.set_ylim(f_min,f_max)
ax1.set_xlim(0.,(h-H)/float(Fs))
ax1.set_title('Spectrogram and peak analysis')
tmp_title='Compare greedy and LP partial tracking on chirps in noise, SNR %d dB' % (int(np.round(D_r)),)
ax2.set_ylim(f_min,f_max)
ax2.set_xlim(0.,(h-H)/float(Fs))
ax2.set_title('Greedy method')
ax3.set_ylim(f_min,f_max)
ax3.set_xlim(0.,(h-H)/float(Fs))
ax3.set_title('LP method')
ax3.set_xlabel("Time in seconds")
ax2.set_ylabel("Frequency in Hz")

fig.savefig(plotoutpath)
with open(plotoutpath[:plotoutpath.rfind('.eps')]+'.txt','w') as f:
    f.write(tmp_title+'%')

if show_plot:
    plt.show()
