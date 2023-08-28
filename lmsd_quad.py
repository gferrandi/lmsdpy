# Python 3.9.12
import csv
import numpy as np
# import scipy.linalg as la
import time
import warnings
# warnings.filterwarnings('error')
# warnings.filterwarnings('ignore')
# warnings.resetwarnings()

from numpy.linalg import norm, cholesky, svd, solve, eig, eigvalsh, eigvals
from scipy.linalg import qr #, cholesky, svd, solve, eigvals 
# from scipy.linalg import solve_continuous_lyapunov

def iter_chol(GG, P):
    while True:
        try:
            R = cholesky(GG[np.ix_(P, P)]).T
            break
        except:
            P = np.delete(P, 0)
    return R, P

def augmented_GG(G, g):
    ll = G.shape[1]+1
    GG = np.zeros((ll,ll))
    GG[:-1,:-1] = G.T.dot(G)
    GG[:-1,-1] = (G.T.dot(g)).reshape(-1)
    GG[-1,:-1] = GG[:-1,-1]
    GG[-1,-1] = np.dot(g,g)
    return(GG)

def wrap_npdiff(A, b): # BB1
    return -np.diff(A, axis = 1, append = b.reshape(-1,1))

def reverse_cumsum(Y, g): # BB2
    Ytmp = Y.copy()
    Ytmp[:,-1] -= g.reshape(-1)
    return np.cumsum(Ytmp[:,::-1], axis = 1)[:,::-1]    

def rayleigh_quotient(x, A):
    return np.dot(x, A.dot(x)) #/np.dot(x,x)
                
def reduced_hessian_quad(G, g, mem, option, stepsize, tol_lin_dep, my_fun):
    if option == 'svd':
        u, s, v = svd(G, full_matrices=False) 
        v = v.transpose()

        # Check singular values and truncate
        P = s > tol_lin_dep*s[0]
        u, s, v = u[:,P], s[P], v[:,P]

        # [Sigma * V.T | U.T @ g] @ J @ (V * Sigma_inv)
        AU = my_fun(v.T * s.reshape(-1,1), u.T.dot(g)) * mem.reshape(1,-1)
        B = AU.dot(v / s.reshape(1,-1))
        B = 0.5*(B + B.T)

    elif option == 'pivoted-qr':
        Q, R, P = la.qr(G, pivoting=True, mode = 'economic')
        R1, P1 = R, np.argsort(P)

        # Check diagonal of R and truncate
        diagr = abs(np.diag(R))
        to_keep = diagr > tol_lin_dep*diagr[0]
        Q, R, P = Q[:,to_keep], R[np.ix_(to_keep, to_keep)], P[to_keep]
        
        # Z = [[R_G R_12] @ Perm_inv | Q.T @ g] @ J @ Perm @ R_G_inv
        # Since Z.T = Z, R_G @ Z = ([[R_G R_12] @ Perm_inv | Q.T @ g] @ J @ Perm).T
        AU = my_fun(R1[:len(P),P1], Q.T.dot(g)) * mem.reshape(1,-1)
        B = np.linalg.solve(R.T, AU[:,P].T)
        B = 0.5*(B + B.transpose())

    elif option == 'chol':
        P = np.array(range(G.shape[1]))
        GG = G.T.dot(G)
        
        # Cholesky factor
        R, P = iter_chol(GG, P)
        
        # [R | r] @ J @ R_inv
        small_r = np.linalg.solve(R.T, G[:,P].T.dot(g)) 
        AU = my_fun(R, small_r) * mem[P].reshape(1,-1) 
        B = solve(R.T, AU.T)
        B = np.triu(B,0) + np.triu(B,1).transpose()

    elif np.logical_or(option == 'cholh', option == 'cholh-ort'):
        P = np.array(range(G.shape[1]+1))
        GG = augmented_GG(G, g)
        
        # Cholesky factor
        R, P = iter_chol(GG, P)
        
        # when we are left with <= 2 gradients, B reduces to the inverse BB2 stepsize!!        
        if P.shape[0] > 2:
            # [R | r  ] @ J @ R_inv
            # [0 | rho]
            AU = -np.diff(R, axis = 1) * mem[P[:-1]].reshape(1,-1) 
            AU = solve(R[:-1,:-1].T, AU.T).T
            
            # cf curtis and guo 2016 
            # B = [R | r  ] @ J @ R_inv, csi = [0 | rho] @ J @ R_inv
            T, csi = AU[:-1,:], AU[-1,:]
            
            # reinforce B's tridiaognal structure
            T = np.tril(T,0) + np.tril(T,-1).T
            
            # Q.T @ A**2 @ Q = T.T @ T + csi @ csi.T
            # find (Q.T @ A @ Q)_inv @ Q.T @ A**2 @ Q
            B = np.linalg.solve(T,  T.T.dot(T) + np.outer(csi, csi))
            
           
            if option == 'cholh-ort':
                # compute harmonic Ritz values and rayleigh quotients
                w, vec1 = eig(B)
                w = np.apply_along_axis(rayleigh_quotient, 0, vec1[:,w>0], T) 
            else:
                w = eigvals(B)
                
            # correct weird values
            w = np.real(w)
            w = w[w > 0]
            return np.sort(1/w)
        else:
            g0, alpha0, y = G[:,-1], 1/mem[-1], g - G[:,-1]
            if option == 'cholh-ort':
                # return BB1
                return np.array([-alpha0*np.dot(g0,g0)/np.dot(g0,y)])
            else:
                # return BB2
                return np.array([-alpha0*np.dot(g0, y)/np.dot(y, y)])
      
    # compute eigenvalues of symmetric matrix; exceptions have already been handled
    w = eigvalsh(B) 
    w = w[w > 0]
    if stepsize == 'bb1':
        steps = np.sort(1/w)
    elif stepsize == 'bb2':
        steps = np.sort(w)
        
    return steps  

def lmsd_quad(p, M2 = 5, stepsize = 'bb1', option = 'svd', tol_lin_dep = 1e-8, tol = 1e-6, maxit = 5e4, alpha = 1):
    
    name_algo = 'LMSD-' + option + '-' + stepsize
    time_start = time.time()
    
    my_fun = wrap_npdiff if stepsize == 'bb1' else reverse_cumsum
    
    # extract info from quadratic problem
    x = p.x0
    f, g = p.obj(x, gradient=True)  # objective and gradient
    f0val, fref = f, f
    gnorm = norm(g)
    tol = tol*gnorm
    
    # init step
    steps, step_len, step_ind = [alpha], 1, 0
    
    # init memory
    G = np.zeros((x.shape[0], M2))
    mem = np.zeros(M2)
    
    # other counters
    neg_eig, nsweep = 0, 0
    
    to_use, itmp, iters = 0, 0, 0
    
    while iters < maxit:
        
        alpha = steps[step_ind]

        # update iterate
        g0, gnorm0, x0, f0 = g, gnorm, x, f
        x = x - alpha*g0
        f, g = p.obj(x, gradient = True)
        
        # check convergence
        gnorm = norm(g)
        if gnorm <= tol:
            break
        
        # Fletcher's method to preserve monotonicity
        if f >= fref:
            # ovverride the current iteration and compute Cauchy; then clear the stack
            x, g, gnorm, f = x0, g0, gnorm0, f0 
            steps, step_len, step_ind, cauchy = gnorm**2/np.dot(g, p.prod(g.reshape(-1,1))), 1, 0, 1
            iters += 1
            continue # ignore what happens next in the loop
        
        # add vectors to memory
        if stepsize == 'bb1':
            G[:,itmp], mem[itmp] = g0, 1/alpha
        elif stepsize == 'bb2':
            G[:,itmp], mem[itmp] = g-g0, alpha
        
        # indices
        itmp = 0 if (itmp > M2-2) else itmp + 1
        to_use = min(M2, to_use + 1)

        # if stack is not clear or the norm of g has increased
        if (step_ind < step_len-1) & (gnorm <= gnorm0): 
            step_ind += 1
        else:
            nsweep += 1
            jtmp = np.roll(list(range(0,M2)), -itmp)[-to_use:]
            
            # compute new steps
            steps = reduced_hessian_quad(G[:,jtmp], g, mem[jtmp], option, stepsize, tol_lin_dep, my_fun)
            step_len, step_ind = len(steps), 0
            
#             if any(np.logical_or(steps < 1/lambda_max, steps > 1/lambda_min)):
#                 print(steps)
            
            # function value at the beginning of next sweep
            fref = f
            
        iters += 1
     
    return {'id_algo': name_algo, 'problem': p.name, 'size': p.n, 'memory':M2, 'nfe':iters+1, 'f0': f0val, 'fstar': f, 'normg': gnorm, 'tol': tol, 'time_elapsed': time.time() - time_start, 'nsweep':nsweep}