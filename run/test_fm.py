'''
'''
import os 
import numpy as np 
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


def test_smlm(): 
    # read in SMLM ground truth
    f_truth = os.path.join(UT.dat_dir(), 'tub1', 'fluoro_truth', 'frames', '02386.csv')
    x, y, z, I = np.loadtxt(f_truth, delimiter=',', skiprows=1, unpack=True, usecols=[2,3,4,5]) # positions in nm
    print('%i fluorophores' % len(x))
    
    # convert to pixels 
    pos = np.zeros((len(x), 2))
    pos[:,0] = x
    pos[:,1] = y 
    pos_pix = FM.pixelize(pos) 
    pos_psf = FM.smlm(pos, I, psf_std=4., noise=2.)

    fig = plt.figure(figsize=(10,10))
    plt.imshow(pos_psf)
    plt.scatter(pos_pix[:,0], pos_pix[:,1], s=100, facecolors='none', edgecolor='C1', linewidths=2)
    fig.savefig(os.path.join(UT.fig_dir(), 'smlm.test.png'), bbox_inches='tight') 
    return None


def test_pixelize(): 
    '''
    '''
    # read in SMLM ground truth
    f_truth = os.path.join(UT.dat_dir(), 'tub1', 'fluoro_truth', 'frames', '02386.csv')
    x, y, z, I = np.loadtxt(f_truth, delimiter=',', skiprows=1, unpack=True, usecols=[2,3,4,5]) # positions in nm
    print('%i fluorophores' % len(x))
    
    # convert to pixels 
    pos = np.zeros((len(x), 2))
    pos[:,0] = x
    pos[:,1] = y 
    pos_pix = FM.pixelize(pos) 

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    sub.scatter(pos_pix[:,0], pos_pix[:,1], marker='x', s=50, color='k')
    sub.scatter(x/150., y/150., s=100, facecolors='none', edgecolor='r', linewidths=2)
    sub.set_xlim(0., 256) 
    sub.set_ylim(0., 256) 
    fig.savefig(os.path.join(UT.fig_dir(), 'pixelize.test.png'), bbox_inches='tight') 
    return None


if __name__=="__main__": 
    test_smlm()
