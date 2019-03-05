'''
'''
import numpy as np 
from astropy.convolution import convolve, Gaussian2DKernel


def smlm(pos, I, psf_std=1., noise=1.): 
    ''' given positions and intensities of fluorophores, produce "realistic" 
    SMLM images. Pixelizes the positions (this needs more thought), applies
    a PSF, then applies simple Gaussian noise. 
    '''
    assert pos.shape[0] == len(I) 
    pos_pix = pixelize(pos) # put fluorophore positions onto a pixel 
    pos_matrix = np.zeros((256, 256)) 
    for x, y, _I in zip(pos_pix[:,0], pos_pix[:,1], I):
        pos_matrix[y,x] = _I 

    # apply PSF 
    gauss_kern = Gaussian2DKernel(psf_std)
    pos_psf = convolve(pos_matrix, gauss_kern)
    
    # add noise 
    pos_psf += noise * np.random.randn(256, 256) 
    return pos_psf 


def pixelize(pos): 
    ''' given x,y positions round the positions and put into pixels
    of 256 x 256 grid. Each pixel is 150nm wide. 

    :param pos: 
        N x 2 array of x and y positions in nm 
    '''
    x, y = pos[:,0], pos[:,1] 
    assert x.max() < 256. * 150.
    assert y.max() < 256. * 150.
    
    xpix, ypix = x/150., y/150.  # nm to pixel 
    
    pos_pix = np.zeros_like(pos).astype(int) 
    pos_pix[:,0] = np.round_(xpix).astype(int)
    pos_pix[:,1] = np.round_(ypix).astype(int)
    return pos_pix
