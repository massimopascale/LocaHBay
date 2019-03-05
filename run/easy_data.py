'''

Generate some easy data to do toy model calculations on 

'''
import os 
import pickle
import scipy as sp 
import numpy as np 
from PIL import Image
from astropy.convolution import convolve, Gaussian2DKernel
from locahbay import fm as FM 
from locahbay import util as UT 
# -- plotting -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def toyInference(i): 
    '''
    '''
    # lets start with toy data
    I0 = 1600.
    noise = 100. 
    toydata = np.zeros((40,40)) 
    toydata[3,3] = I0 
    toydata[35,23] = I0 
    toydata[23,20] = I0 
    gauss_kern = Gaussian2DKernel(0.5)
    toydata = convolve(toydata, gauss_kern) # apply PSF 
    np.random.seed(1)
    toydata += noise * np.random.randn(toydata.shape[0], toydata.shape[1]) # add noise 

    f_density = 1./float(toydata.shape[0] * toydata.shape[1])
    print("f_density = %f" % f_density) 
    
    def psfmodel(tt): 
        pos = tt.reshape(toydata.shape)
        return convolve(pos, gauss_kern)
    
    def lnLike(tt): 
        return np.sum(0.5*(toydata - psfmodel(tt))**2/noise**2)

    from scipy.stats import norm as Norm

    def p1(_I): 
        # the intensity function is uniform between 10 and 20 
        _I = np.atleast_1d(_I)
        _p1 = np.zeros_like(_I) 
        _p1[(_I > 0.) & (_I < 2000.)] = 0.05
        return _p1

    def p2(_I): 
        _p2 = lambda ii: p1(_I - ii) * p1(ii) 
        return sp.integrate.quad(_p2, -np.inf, np.inf)[0]

    def lnPrior(tt): 
        p1tt = np.array([p1(ttt) for ttt in tt])
        #p2tt = np.array([p2(ttt) for ttt in tt])

        pri = np.sum(np.log((1.-f_density) * Norm.pdf(np.log(tt), loc=-10, scale=1e-2)/tt + 
            f_density * p1tt)) # ignore the second order term 
        #pri = np.sum(np.log( 
        #    (1.-f_density-f_density**2) * Norm.pdf(np.log(tt), loc=-10, scale=1)/tt + 
        #    f_density * p1tt + f_density**2 * p2tt))
        #print -pri 

        if np.isfinite(pri): 
            return -pri
        else: 
            return np.inf  

    def lnPost(tt): 
        return lnLike(tt) + lnPrior(tt)

    tt0 = np.repeat(0.5*I0, toydata.size) 
    print lnPost(tt0)
    min_result = sp.optimize.minimize(lnPost, tt0, method='BFGS', options={"eps": 0.1, "maxiter":10000})
    print lnPost(min_result['x'])
    print min_result['x'][33]

    fig = plt.figure(figsize=(15,5))
    plt.subplot(131) 
    plt.imshow(toydata, vmin=0., vmax=1e3)
    plt.title('Toy Data') 
    plt.subplot(132) 
    plt.imshow(psfmodel(min_result['x']), vmin=0., vmax=1e3)
    plt.title('Model w/ PSF') 
    plt.subplot(133) 
    plt.imshow(min_result['x'].reshape(toydata.shape), vmin=0., vmax=1e3)
    plt.title('Model') 
    fig.savefig(os.path.join(UT.fig_dir(), 'toyinference.png'), bbox_inches='tight') 
    return None 


def easySMLM(i): 
    ''' way simplified SMLM dataset where the intensities of the fluorophores 
    are all 1500. The PSF is 2 pixels (300nm). And there is onlly Gaussian noise 
    of sigma=100
    '''
    f_truth = os.path.join(UT.dat_dir(), 'tub1', 'fluoro_truth', 'frames', str(i).zfill(5)+'.csv') 
    x, y = np.loadtxt(f_truth, delimiter=',', skiprows=1, unpack=True, usecols=[2,3]) # positions in nm
    
    # convert to pixels 
    pos = np.zeros((len(x), 2))
    pos[:,0] = x
    pos[:,1] = y 
    pos_pix = FM.pixelize(pos) 
    pos_psf = FM.smlm(pos, np.repeat(1500., len(x)), psf_std=2., noise=100.)

    # save "image" to file 
    pickle.dump(pos_psf, open(os.path.join(UT.dat_dir(), 'tub1', 'easy_data', 'easy.%i.p' % i), 'wb'))

    fig = plt.figure(figsize=(10,10))
    plt.imshow(pos_psf)
    plt.scatter(pos_pix[:,0], pos_pix[:,1], s=100, facecolors='none', edgecolor='C1', linewidths=2)
    fig.savefig(os.path.join(UT.fig_dir(), 'easy.smlm.%i.png' % i), bbox_inches='tight') 
    return None


def intensities(): 
    ''' look at the intensity distribution of the fluorophores  
    average fluorophore intensities = 1565.9
    '''
    f_truth = os.path.join(UT.dat_dir(), 'tub1', 'fluoro_truth', 'activation.csv')
    intensity = np.loadtxt(f_truth, skiprows=1, delimiter=',', usecols=[5]) 
    print('average fluorophore intensities = %.1f' % np.average(intensity))

    fig = plt.figure(figsize=(8,4))
    sub = fig.add_subplot(111)
    sub.hist(intensity, range=(0, 4000), bins=40) 
    sub.vlines(np.average(intensity), 0., 8000.) 
    sub.set_xlabel('Fluorophore Intensities', fontsize=20) 
    sub.set_xlim(0, 4000) 
    sub.set_ylim(0, 7000) 
    fig.savefig(os.path.join(UT.fig_dir(), 'fluorophore.intensity.png'), bbox_inches='tight') 
    return None


def noise(i): 
    ''' mask out of the fluorophores and see what the noise is like
    '''
    f_image = os.path.join(UT.dat_dir(), 'tub1', str(i).zfill(5)+'.tif') 
    im = Image.open(f_image)
    imarr = np.array(im)

    f_truth = os.path.join(UT.dat_dir(), 'tub1', 'fluoro_truth', 'frames', str(i).zfill(5)+'.csv') 
    x, y = np.loadtxt(f_truth, delimiter=',', skiprows=1, unpack=True, usecols=[2,3]) # positions in nm
    xpix, ypix = np.round_(x/150.).astype(int), np.round_(y/150.).astype(int)
    
    # mask out fluorophors
    truth_mask = np.zeros_like(imarr).astype(bool) 
    for x, y in zip(np.round(xpix), np.round(ypix)):
        for yoffset in range(-5, 5): 
            for xoffset in range(-5, 5): 
                truth_mask[y+yoffset, x+xoffset] = True 
    imarr[truth_mask] = 0. 
    fig = plt.figure(figsize=(10,10))
    plt.imshow(imarr)
    fig.savefig(os.path.join(UT.fig_dir(), 'noise.%i.test.png' % i), bbox_inches='tight') 
    
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    sub.hist(imarr[~truth_mask].flatten(), range=(0,600), bins=50)
    sub.set_xlim(0., 600) 
    fig.savefig(os.path.join(UT.fig_dir(), 'noise.%i.hist.test.png' % i), bbox_inches='tight') 
    return None


if __name__=="__main__": 
    #intensities()
    #noise(1)
    #for i in range(1, 101):
    #    easySMLM(i)
    toyInference(1)
