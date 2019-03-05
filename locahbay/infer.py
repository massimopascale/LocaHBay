'''
inference  
'''

def lnLike(m, d, sig): 
    ''' log likelihood
    ''' 
    return np.sum(0.5 * (m - d)**2/sig**2) 


def lnPrior_easy(): 
