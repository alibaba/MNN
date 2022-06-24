from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar

def VBMF(Y, cacb, sigma2=None, H=None):
    """Implementation of the analytical solution to Variational Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to VBMF. 
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.
        To estimate cacb, use the function EVBMF().

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.
        
    cacb : int
        Product of the prior variances of the matrices that factorize the input.
    
    sigma2 : int or None (default=None)
        Variance of the noise on Y.
        
    H : int or None (default = None)
        Maximum rank of the factorized matrices.
        
    Returns
    -------
    U : numpy-array
        Left-singular vectors. 
        
    S : numpy-array
        Diagonal matrix of singular values.
        
    V : numpy-array
        Right-singular vectors.
        
    post : dictionary
        Dictionary containing the computed posterior values.
        
        
    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
    
    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.
    """    
    
    L,M = Y.shape #has to be L<=M

    if H is None:
        H = L
    
    #SVD of the input matrix, max rank of H
    U,s,V = np.linalg.svd(Y)
    U = U[:,:H]
    s = s[:H]
    V = V[:H].T 

    #Calculate residual
    residual = 0.
    if H<L:
        residual = np.sum(np.sum(Y**2)-np.sum(s**2))

    #Estimation of the variance when sigma2 is unspecified
    if sigma2 is None: 
        upper_bound = (np.sum(s**2)+ residual)/(L+M)

        if L==H: 
            lower_bound = s[-1]**2/M
        else:
            lower_bound = residual/((L-H)*M)

        sigma2_opt = minimize_scalar(VBsigma2, args=(L,M,cacb,s,residual), bounds=[lower_bound, upper_bound], method='Bounded')
        sigma2 = sigma2_opt.x
        # print("Estimated sigma2: ", sigma2)

    #Threshold gamma term
    #Formula above (21) from [1]
    thresh_term = (L+M + sigma2/cacb**2)/2 
    threshold = np.sqrt( sigma2 * (thresh_term + np.sqrt(thresh_term**2 - L*M) ))
              
    #Number of singular values where gamma>threshold
    pos = np.sum(s>threshold)

    #Formula (10) from [2]
    d = np.multiply(s[:pos], 
                    1 - np.multiply(sigma2/(2*s[:pos]**2),
                                    L+M+np.sqrt( (M-L)**2 + 4*s[:pos]**2/cacb**2 )))

    #Computation of the posterior
    post = {}
    zeta = sigma2/(2*L*M) * (L+M+sigma2/cacb**2 - np.sqrt((L+M+sigma2/cacb**2)**2 - 4*L*M))
    post['ma'] = np.zeros(H) 
    post['mb'] = np.zeros(H)
    post['sa2'] = cacb * (1-L*zeta/sigma2) * np.ones(H)
    post['sb2'] = cacb * (1-M*zeta/sigma2) * np.ones(H)  

    delta = cacb/sigma2 * (s[:pos]-d- L*sigma2/s[:pos])
    post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    post['sa2'][:pos] = np.divide(sigma2*delta, s[:pos])
    post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post['sigma2'] = sigma2
    post['F'] = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 - (L+M)*H
               + np.sum(M*np.log(cacb/post['sa2']) + L*np.log(cacb/post['sb2'])
                        + (post['ma']**2 + M*post['sa2'])/cacb + (post['mb']**2 + L*post['sb2'])/cacb
                        + (-2 * np.multiply(np.multiply(post['ma'], post['mb']), s)
                           + np.multiply(post['ma']**2 + M*post['sa2'],post['mb']**2 + L*post['sb2']))/sigma2))

    return U[:,:pos], np.diag(d), V[:,:pos], post


def VBsigma2(sigma2,L,M,cacb,s,residual):
    H = len(s)

    thresh_term = (L+M + sigma2/cacb**2)/2 
    threshold = np.sqrt( sigma2 * (thresh_term + np.sqrt(thresh_term**2 - L*M) ))
    pos = np.sum(s>threshold)
    
    d = np.multiply(s[:pos], 
                    1 - np.multiply(sigma2/(2*s[:pos]**2),
                                    L+M+np.sqrt( (M-L)**2 + 4*s[:pos]**2/cacb**2 )))

    zeta = sigma2/(2*L*M) * (L+M+sigma2/cacb**2 - np.sqrt((L+M+sigma2/cacb**2)**2 - 4*L*M))
    post_ma = np.zeros(H) 
    post_mb = np.zeros(H)
    post_sa2 = cacb * (1-L*zeta/sigma2) * np.ones(H)
    post_sb2 = cacb * (1-M*zeta/sigma2) * np.ones(H)  

    delta = cacb/sigma2 * (s[:pos]-d- L*sigma2/s[:pos])
    post_ma[:pos] = np.sqrt(np.multiply(d, delta))
    post_mb[:pos] = np.sqrt(np.divide(d, delta))
    post_sa2[:pos] = np.divide(sigma2*delta, s[:pos])
    post_sb2[:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))

    F = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 - (L+M)*H
               + np.sum(M*np.log(cacb/post_sa2) + L*np.log(cacb/post_sb2)
                        + (post_ma**2 + M*post_sa2)/cacb + (post_mb**2 + L*post_sb2)/cacb
                        + (-2 * np.multiply(np.multiply(post_ma, post_mb), s)
                           + np.multiply(post_ma**2 + M*post_sa2,post_mb**2 + L*post_sb2))/sigma2))
    return F



def EVBMF(Y, sigma2=None, H=None):
    """Implementation of the analytical solution to Empirical Variational Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to empirical VBMF. 
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.
    
    sigma2 : int or None (default=None)
        Variance of the noise on Y.
        
    H : int or None (default = None)
        Maximum rank of the factorized matrices.
        
    Returns
    -------
    U : numpy-array
        Left-singular vectors. 
        
    S : numpy-array
        Diagonal matrix of singular values.
        
    V : numpy-array
        Right-singular vectors.
        
    post : dictionary
        Dictionary containing the computed posterior values.
        
        
    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
    
    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.     
    """   
    L,M = Y.shape #has to be L<=M

    if H is None:
        H = L

    alpha = L/M
    tauubar = 2.5129*np.sqrt(alpha)
    
    #SVD of the input matrix, max rank of H
    U,s,V = np.linalg.svd(Y)
    U = U[:,:H]
    s = s[:H]
    V = V[:H].T 

    #Calculate residual
    residual = 0.
    if H<L:
        residual = np.sum(np.sum(Y**2)-np.sum(s**2))

    #Estimation of the variance when sigma2 is unspecified
    if sigma2 is None: 
        xubar = (1+tauubar)*(1+alpha/tauubar)
        eH_ub = int(np.min([np.ceil(L/(1+alpha))-1, H]))-1
        upper_bound = (np.sum(s**2)+residual)/(L*M)
        lower_bound = np.max([s[eH_ub+1]**2/(M*xubar), np.mean(s[eH_ub+1:]**2)/M])

        scale = 1.#/lower_bound
        s = s*np.sqrt(scale)
        residual = residual*scale
        lower_bound = lower_bound*scale
        upper_bound = upper_bound*scale

        sigma2_opt = minimize_scalar(EVBsigma2, args=(L,M,s,residual,xubar), bounds=[lower_bound, upper_bound], method='Bounded')
        sigma2 = sigma2_opt.x

        # print(sigma2)

    #Threshold gamma term
    threshold = np.sqrt(M*sigma2*(1+tauubar)*(1+alpha/tauubar))
    pos = np.sum(s>threshold)

    #Formula (15) from [2]
    d = np.multiply(s[:pos]/2, 1-np.divide((L+M)*sigma2, s[:pos]**2) + np.sqrt((1-np.divide((L+M)*sigma2, s[:pos]**2))**2 -4*L*M*sigma2**2/s[:pos]**4) )

    #Computation of the posterior
    post = {}
    post['ma'] = np.zeros(H) 
    post['mb'] = np.zeros(H)
    post['sa2'] = np.zeros(H) 
    post['sb2'] = np.zeros(H) 
    post['cacb'] = np.zeros(H)  

    tau = np.multiply(d, s[:pos])/(M*sigma2)
    delta = np.multiply(np.sqrt(np.divide(M*d, L*s[:pos])), 1+alpha/tau)

    post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    post['sa2'][:pos] = np.divide(sigma2*delta, s[:pos])
    post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos])/(L*M))
    post['sigma2'] = sigma2
    post['F'] = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 
                     + np.sum(M*np.log(tau+1) + L*np.log(tau/alpha +1) - M*tau))

    return U[:,:pos], np.diag(d), V[:,:pos], post

def EVBsigma2(sigma2,L,M,s,residual,xubar):
    H = len(s)

    alpha = L/M
    x = s**2/(M*sigma2) 

    z1 = x[x>xubar]
    z2 = x[x<=xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum( np.log( np.divide(tau_z1+1, z1)))
    term4 = alpha*np.sum(np.log(tau_z1/alpha+1))
    
    obj = term1+term2+term3+term4+ residual/(M*sigma2) + (L-H)*np.log(sigma2)

    return obj

def phi0(x):
    return x-np.log(x)

def phi1(x, alpha):
    return np.log(tau(x,alpha)+1) + alpha*np.log(tau(x,alpha)/alpha + 1) - tau(x,alpha)

def tau(x, alpha):
    return 0.5 * (x-(1+alpha) + np.sqrt((x-(1+alpha))**2 - 4*alpha))
