import matplotlib.pyplot as plt
import numpy as np
import bestpathslp as bplp

fname='/Users/mondaugen/Documents/sound/timit/TIMIT/TEST/DR1/FAKS0/SA1_your_dark.f64'
x=np.fromfile(fname,dtype='float64')
Fs=16000
H=256
a=bplp.estimate_ddm_decomp(x,H=H,th_ddm=0.,#10.**(-100./20),
        Fs=Fs)
fig,ax1=plt.subplots(1,1)
h=0
for a_ in a:
    for a__ in a_:
        f0_=np.imag(a__[1])/(2.*np.pi)*Fs
        f1_=(np.imag(a__[1])+2.*np.imag(a__[2])*H)/(2.*np.pi)*Fs
        t0_=h/float(Fs)
        t1_=(h+H)/float(Fs)
        ax1.plot([t0_,t1_],[f0_,f1_],c='k')
    h+=H
plt.show()
