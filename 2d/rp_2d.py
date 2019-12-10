

import autograd as Agrad
import autograd.numpy as np 
import scipy.optimize
import scipy.stats as st
from scipy.integrate import trapz
from scipy.integrate import simps
from photutils import find_peaks
from photutils import detect_threshold
# -- plotting --- 
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


########################################################################
'''
rp_2d.py
Code modified from rp_2d.py
Author: Massimo Pascale
Last Updated: 11/12/2019

Code uses poisson prior and exponential intensity function to determine
point source locations in psf+noise and recover hyperparameters.
'''
########################################################################




#create global definitions - this will become a main function later on
np.random.seed(42)
Ndata = 5;
n_grid = 16;
pix_1d = np.linspace(0., 1., n_grid) # pixel gridding
fdensity_true = float(Ndata)/float(n_grid**2); #number density of obj in 1d

sig_psf = 0.1 # psf width
sig_noise = 0.02 # noise level

#these are values for the power law function for sampling intensities
w_interval = (1,2);
w_lin = np.linspace(1,2,100);
alpha = 2;
w_norm = (50**(alpha+1) - w_interval[0]**(alpha+1))/(alpha+1);
w_func = np.power(w_lin,alpha)/w_norm;
w_true = w_norm*np.random.choice(w_func,Ndata);


def psi(pos): 
    ''' measurement model, which in our case is just a 1d gaussian of width 
    sigma (PSF) written out to a meshgrid created by pix1d 
    '''
    x,y = np.meshgrid(pix_1d,pix_1d);
    return np.exp(-((y-pix_1d[pos[0]])**2 + (x - pix_1d[pos[1]])**2)/2/sig_psf**2); #keep in mind difference between x and y position and indices! Here, you are given indices, but meshgrid is in x-y coords

def gaussian(x, loc=None, scale=None): 
    '''
    scipy's gaussian pdf didn't work idk
    '''
    y = (x - loc)/scale
    return np.exp(-0.5*y**2)/np.sqrt(2.*np.pi)/scale

    
def Psi(ws): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    return np.sum(np.array([w*psi(index) for (index,w) in np.ndenumerate(ws)]),0)

def prior_i(w,fdensity,alpha,sig):
    '''
    log of Poisson prior for an indivudial pixel
    '''
    pri=0.;
    if 0. < w <= 4:
        #norm = (max(interval_grid)**(alpha+1) - min(interval_grid)**(alpha+1))/(alpha+1); #normalization of mass function
        p1 = w**alpha /w_norm; #probability of single source
        w_fft = np.linspace(0,w,50);
        #now probability of second source
        p2 = np.abs(Agrad.numpy.fft.ifft(Agrad.numpy.fft.fft(w_fft**alpha /w_norm)**2));
        p2 = p2[-1];
        pri += fdensity*p1 + p2*fdensity**2
    if w > 0:
        pri += (1.-fdensity - fdensity**2 ) *gaussian(np.log(w),loc=-4., scale=sig)/w
    return pri
    
def lnprior(ws,fdensity,alpha,sig): 
	'''
	calculate log of prior
	'''
	return np.sum([np.log(prior_i(w,fdensity,alpha,sig)) for w in ws.flatten()])

def lnlike(ws): 
    ''' log likelihood 
    '''
    return -0.5 * np.sum((Psi(ws) - data)**2/sig_noise**2)
 
def lnpost(ws,fdensity,alpha,sig): 
    #converting flattened ws to matrix
    ws = ws.reshape((n_grid,n_grid));
    return lnlike(ws) + lnprior(ws,fdensity,alpha,sig)

########################################################################
#create mock data to run on
########################################################################
#create coordinate grid
theta_grid = np.linspace(0., 1., n_grid) # gridding of theta (same as pixels)

#create true values - assign to grid
x_true = np.abs(np.random.rand(Ndata)) # location of sources
y_true = np.abs(np.random.rand(Ndata));

#w_true = np.abs(np.random.rand(Ndata))+1;

#true grid needs to be set up with noise
w_true_grid = np.zeros((n_grid,n_grid))
for x,y, w in zip(x_true,y_true, w_true): 
    w_true_grid[np.argmin(np.abs(theta_grid - x)),np.argmin(np.abs(theta_grid - y))] = w
data = Psi(w_true_grid) + sig_noise * np.random.randn(n_grid,n_grid);

########################################################################
#now begin the actual execution
########################################################################


#now we begin the optimization
tt0 = np.zeros(n_grid**2) +1; #begin with high uniform M

#begin with the simple method of just minimizing
f_curr = fdensity_true;
alpha_curr = 2;
sig_delta = 0.75;
#keeping in mind that minimize requires flattened arrays
res = scipy.optimize.minimize(Agrad.value_and_grad(lambda tt: -1.*lnpost(tt,f_curr,alpha_curr,sig_delta)),
                              tt0, # theta initial
                              jac=True, 
                              method='L-BFGS-B', 
                              bounds=[(1e-5, 10)]*len(tt0))
                              
tt_prime = res['x'];
w_final = tt_prime.reshape((n_grid,n_grid));
#pick out the peaks using photutils
thresh = detect_threshold(w_final,3);
tbl = find_peaks(w_final,thresh);
positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
w_peaks = np.zeros((n_grid,n_grid));
w_peaks[positions] = w_final[positions];



fig, ax = plt.subplots(1,3)
ax[0].imshow(w_true_grid);
ax[0].set_title('True Positions')
ax[1].imshow(data);
ax[1].set_title('Observed Data')
ax[2].imshow(w_peaks);
ax[2].set_title('Sparse Bayes')
plt.show();

'''
f_curr = 0.3;
alpha_curr = 0.2;
sig_delta = 0.75;
step = 0;
#now we begin optimizing step by step, beginning with M, then f, then alpha
while(True):
	#start with m
	res = scipy.optimize.minimize(
			Agrad.value_and_grad(lambda tt: -1.*lnpost(tt,f_curr,alpha_curr,sig_delta,w_grid)),  
			tt0, # theta initial 
			jac=True, 
			method='L-BFGS-B', 
			bounds=[(1e-5, 5)]*len(tt0))
	tt_prime = res['x']
	print('midstep');
	#step f by clipping the data
	m_inds = scipy.signal.find_peaks(tt_prime);
	f_prime = len(m_inds)/n_grid;
	tt_prime = tt_prime[m_inds];
	#step f and alpha simultaneously
	res = scipy.optimize.minimize(
			Agrad.value_and_grad(lambda x: -1.*lnpost(tt_prime,x,sig_delta,w_grid)),  
			(alpha_curr), # fdensity and alpha initial 
			jac=True, 
			method='L-BFGS-B', 
			bounds=[(-10, 10)])
	f_prime,alpha_prime = res['x']
	#calculate difference for error threshold
	diff_f = abs(f_prime-f_curr);
	diff_alpha = abs(alpha_prime - alpha_curr);
	diff_t = abs(tt_prime - tt0);
	diff_w = np.sqrt(diff_t.dot(diff_t));
	if diff_w+diff_f+diff_alpha < 1e-4:
		break;
	#if not passed, then update values and step
	tt0 = tt_prime;
	alpha_curr = alpha_prime;
	f_curr = f_prime;
	step+=1;
	print(step);

print(alpha_prime);
print(f_prime);
'''
'''
# plot data 
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
sub.scatter(xpix, data, marker='x', s=10) 
for x, w in zip(x_true, w_true): 
    sub.vlines(x, 0, w, color='k')
sub.plot(theta_grid, tt_prime, c='C1', ls='--')
sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
sub.set_xlim(0., 1.) 
sub.set_ylabel('intensity', fontsize=25) 
sub.set_ylim(0., 2.5) 
plt.show();
'''