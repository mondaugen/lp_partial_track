# compare with MQ algorithm
# try in different SNRs to see performance
import matplotlib.pyplot as plt
import numpy as np
import bestpathslp as bplp
import cvxopt
import ddm

plt.rc('text',usetex=True)
plt.rc('font',**ddm.FONT_OPT['dafx'])

show_plot=False

#fname='/Users/mondaugen/Documents/sound/timit/TIMIT/TEST/DR1/FAKS0/SA1_your_dark.f64'
fname='sounds/chirps_simple.f64'
# minimum power of atom to consider
minpow=-60
SNRs=[0,-6,-12]
fig,axs=plt.subplots(3,len(SNRs),sharex=True,sharey=True)
# Boundaries on frequencies to consider
f_min=250.
f_max=2000
Fs=16000
H=512
M=2048
y=np.fromfile(fname,dtype='float64')
sigpow=np.sum(y*y)/len(y)
print 'sigpow: %f' % (np.log(sigpow)/np.log(10)*10,)
t_min=0.
t_max=(((len(y)-M)/H)*H)/float(Fs)
# make colours
cmap=plt.get_cmap('magma')
cmapgrey=plt.get_cmap('gray')
for SNR,axr,ax_i in zip(SNRs,axs.T,xrange(axs.shape[1])):
    ax1=axr[1]
    ax2=axr[2]
    ax3=axr[0]

    for i,a in enumerate(['a','b','c']):
        axr[i].set_title('%d.%s.' % (ax_i+1,a))
        axr[i].locator_params(nbins=5)

    noisepow=sigpow*np.power(10.,-SNR/10.)
    print 'noisepow: %f' % (np.log(noisepow)/np.log(10)*10,)
    # add noise
    x=y+np.random.standard_normal(len(y))*np.sqrt(noisepow)
    totpow=np.sum(x*x)/len(x)
    print 'totpow: %f' % (np.log(totpow)/np.log(10)*10,)
    x=x[:len(x)]
    a=bplp.estimate_ddm_decomp(x,
            M=M,
            H=H,
            th_ddm=10.**(minpow/20),
            b_ddm_hz=100.,
            o_ddm_hz=50.,
            Fs=Fs)

    print 'K=%d' % (len(a),)

    # Number of paths L
    L=3


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

    # determine levels of most and least powerful atom
    _minpow=float('inf')
    _maxpow=float('-inf')
    for a_ in a_flt:
        for a__ in a_:
            # compute power
            a_re=np.real(a__)
            n_=np.arange(M)
            pw=np.sum(np.exp(2*(a_re[0]+a_re[1]*n_+a_re[2]*(n_**2.))))/M
            # power in dB 
            pwdb=np.log(pw)/np.log(10)*10.
            if (pwdb > _maxpow):
                _maxpow=pwdb
            if (pwdb < _minpow):
                _minpow=pwdb

    for a_ in a_flt:
        for a__ in a_:
            # compute power
            a_re=np.real(a__)
            n_=np.arange(M)
            pw=np.sum(np.exp(2*(a_re[0]+a_re[1]*n_+a_re[2]*(n_**2.))))/M
            # power in dB 
            pwdb=np.log(pw)/np.log(10)*10.
            # shade of grey based on power
            pwc=(pwdb-_minpow)/(_maxpow-_minpow)
            #print 'pwdb: %f pwc: %f' % (pwdb,pwc)
            c_=cmapgrey(1-pwc)
            f0_=np.imag(a__[1])/(2.*np.pi)*Fs
            f1_=(np.imag(a__[1])+2.*np.imag(a__[2])*H)/(2.*np.pi)*Fs
            t0_=h/float(Fs)
            t1_=(h+H)/float(Fs)
            ax1.plot([t0_,t1_],[f0_,f1_],c=c_)
            ax2.plot([t0_,t1_],[f0_,f1_],c=c_)
            ax3.plot([t0_,t1_],[f0_,f1_],c=c_)
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
        return (abs((np.imag(a[1])+2.*np.imag(a[2])*H)
                -np.imag(b[1])) #+ abs(np.imag(a[2]) - np.imag(b[2]))
                )

    # do MQ algorithm
    # only do sets of K frames or else algorithm intractable

    K=3
    for k in xrange(0,len(i_frames),K-1):
        (sl,D_)=bplp.get_mq_sol_small(i_frames[k:k+K],
                                i_vals,
                                dfunc,
                                0.1,
                                L)
        pths_=[[(i,j) for i,j in zip(sl_[:-1],sl_[1:])] for sl_ in sl]
        for i,pth in enumerate(pths_):
            clr=cmap(float(i+1)/(L+1))
            for pr in pth:
                j,k=pr
                f0_=np.imag(i_vals[j][1])/(2.*np.pi)*Fs
                f1_=np.imag(i_vals[k][1])/(2.*np.pi)*Fs
                t0_=i_times[j]/float(Fs)
                t1_=i_times[k]/float(Fs)
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

    ndp=sum([len(x) for x in a_flt])
    print 'number of data points %d' % (ndp,)
    print 'average number per frame %d' % (ndp/len(a_flt),)
    print 'number of frames %d' % (len(a_flt),)

    (c,G,h,A,b,M_,i_pairs,i_costs)=bplp.get_lp_mats(i_frames,
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

    (pths,i_pths)=bplp.extract_paths(sol['x'],i_pairs)

    for i,pth,i_pth in zip(xrange(len(pths)),pths,i_pths):
        clr=cmap(float(i+1)/(L+1))
        do_label=True
        for pr in pth:
            j,k=pr
            f0_=np.imag(i_vals[j][1])/(2.*np.pi)*Fs
            f1_=np.imag(i_vals[k][1])/(2.*np.pi)*Fs
            t0_=i_times[j]/float(Fs)
            t1_=i_times[k]/float(Fs)
            if (do_label):
                ax1.plot([t0_,t1_],[f0_,f1_],c=clr,lw=2.,label='Chirp %d' %(i+1,))
                do_label=False
            else:
                ax1.plot([t0_,t1_],[f0_,f1_],c=clr,lw=2.)

for i in xrange(3):
    axs[i,0].set_ylabel('Freq. (Hz)')
    axs[i,0].set_ylim([f_min,f_max])
    axs[i,0].set_xlim([t_min,t_max])

for i,s in enumerate(SNRs):
    xlab=axs[-1,i].set_xlabel('Time (s)')

han,lab=axs[1,-1].get_legend_handles_labels()
axs[0,-1].legend(han,lab,handlelength=1.,
        fontsize=xlab.get_fontproperties().get_size_in_points())

fig.suptitle(
    'Comparison of LP and McAulay-Quatieri methods on chirps in noise',
    # hacky way to get title size
    fontsize=fig.get_axes()[0].title.get_fontproperties().get_size_in_points())


#fig2.tight_layout()
plt.subplots_adjust(top=.92)
fig.set_size_inches(10,7)

plt.savefig('paper/plots/mq_lp_compare_chirps.eps')

if (show_plot):
    plt.show()
