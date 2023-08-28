# Python 3.9.12
import csv
import numpy as np
# import scipy.linalg as la
import time
import warnings
# warnings.filterwarnings('error')
warnings.filterwarnings('ignore')
# warnings.resetwarnings()

from numpy.linalg import norm, cholesky, svd, solve, eigvals
from scipy.linalg import qr #, cholesky, svd, solve, eigvals 
# from scipy.linalg import solve_continuous_lyapunov

def custom_norm(g):
#     return np.max(abs(g))
    return norm(g)

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

def sy_for_cholesky(R, alphas):
    return -np.diff(R[:,:-1].T.dot(R), axis = 1) / alphas.reshape(-1,1)

def y_correction(R, alphas):
    sy = sy_for_cholesky(R, alphas)
    L = np.tril(-sy.T + sy) # minus sign is already in the expression
    Z = (L.T * alphas.reshape(1,-1)) * alphas.reshape(-1,1) # DL'D
    R = R[:,:-1]
    ZR = solve(R.T, Z)
    return solve(R.T, ZR.T).T

def wrap_npdiff(A, b):
    return -np.diff(A, axis = 1, append = b.reshape(-1,1))

def lyapunov(a, q, tol_lin_dep, deco = False):
    # please remember to pass only half of matrix as a
    if deco:
        _, d, vt = svd(a, full_matrices=False) # SVD of S
        
        # Check singular values and truncate
        ss = d**2
        P = ss > tol_lin_dep*(ss[0])
        vt, ss = vt[P,:], ss[P]
        
        return vt.dot(q.dot(vt.T))/np.add.outer(ss, ss)
    else:
        # a must be a vector
        return q/np.add.outer(a,a)
        
#     return solve_continuous_lyapunov(a, q)

def reduced_hessian(G, g, gnorm, mem, option, stepsize, tol_lin_dep):
    
    if option in ['fletcher', 'fletcher-pert', 'lya']:
        # compute Cholesky factor for [G g]
        P = np.array(range(G.shape[1]+1))
        GG = augmented_GG(G, g)

        # Cholesky factor
        R, P = iter_chol(GG, P)

        if P.shape[0] == 1:
            # compute BB steps
            y = g - G[:,-1]
            if stepsize == 'bb1':
                B = np.array([[-mem[-1]*np.dot(G[:,-1], y)/np.dot(G[:,-1], G[:,-1])]])
            else:
                B = np.array([[-mem[-1]*np.dot(G[:,-1], y)/np.dot(y, y)]]) # already the BB2, not its reciprocal
        else:
            sel_mem = mem[P[:-1]].reshape(1,-1)
            
            if option in ['fletcher', 'fletcher-pert']:
                AU = -np.diff(R, axis = 1) * sel_mem 
                AU = (solve(R[:-1,:-1].T, AU.T)).T # This gives upper Hessenberg!!
                B, csi = AU[:-1,:], AU[-1,:] # cf curtis and guo 2016 

                if option == 'fletcher-pert': # ONLY FOR BB1 ATM
                    B += y_correction(R[:-1,:], sel_mem)
                    B = 0.5*(B + B.T)
                else:
                    # both bb1 and bb2
                    B = np.tril(B,0) + np.tril(B,-1).T
                    
                if stepsize == 'bb2':
                    B = solve(B.T.dot(B) + np.outer(csi, csi), B)

            elif option == 'lya':
                sy = sy_for_cholesky(R[:-1,:], sel_mem) 
                s = R[:-1,:-1] / sel_mem if stepsize == 'bb1' else np.diff(R, axis = 1)
                B = lyapunov(s, sy.T + sy, tol_lin_dep, deco = True)
#                 B = lyapunov(s.T.dot(s), sy.T + sy, tol_lin_dep, deco = True)
                B = 0.5*(B + B.T)
#                 print('shape =', B.shape, 'norm', norm(B))
                
    elif option == 'lya-svd': # BB1 ONLY
        u, d, v = svd(-G / mem.reshape(1, -1), full_matrices=False) # SVD of S
        v = v.T

        # Check singular values and truncate
        ss = d**2
        P = ss > tol_lin_dep*(ss[0]) # !!!!
#         P = d > tol_lin_dep*(d[0]) 
        u, d, ss, v = u[:,P], d[P], ss[P], v[:,P]
        
        small_g = u.T.dot(g)
        sy = -wrap_npdiff(-(v.T * d.reshape(-1,1)) * mem.reshape(1,-1), small_g) * d.reshape(-1,1)
        sy = sy.dot(v)
        B = lyapunov(ss, sy.T + sy, tol_lin_dep, deco = False)
#         B = lyapunov(np.diag(ss), sy.T + sy, tol_lin_dep, deco = False)
        B = 0.5*(B + B.T)
        
    elif option == 'lya-qr': # BB1 ONLY
        Q, R, P = qr(G, pivoting=True, mode = 'economic')
        R1, P1 = R, np.argsort(P)

        # Check diagonal of R and truncate
        diagr = abs(np.diag(R))
        to_keep = diagr > tol_lin_dep*(diagr[0])
        Q, R, P = Q[:,to_keep], R[np.ix_(to_keep, to_keep)], P[to_keep]

        small_g = G[:,P].T.dot(g)
#         print((R.T @ R1[:len(P),P1]).shape)
        sy = wrap_npdiff(R.T.dot(R1[:len(P),P1]), small_g)[:,P] / mem[P].reshape(-1,1)
        s = R / mem[P].reshape(1,-1)
        B = lyapunov(s, sy.T + sy, tol_lin_dep, deco = True)
#         B = lyapunov(s.T.dot(s), sy.T + sy, tol_lin_dep, deco = True)
        B = 0.5*(B + B.T)
    
    # compute eigenvalues 
    w = eigvals(B) 
    w = np.real(w)
#     print('eigenvalues =', w)
    
    if all(w <= 0):
        steps = 1/np.array([max(1e-5, min(gnorm, 1))])
    else:
        w = w[w > 0]
        
        # sort based on type of stepsize
        steps = np.sort(1/w) if stepsize == 'bb1' else np.sort(w)
    
    return steps
        
# line search
def line_search_lmsd(p, x, alpha, g, fref, gnorm, nfe, nge, quadratic = False):
    flag = 0
    if quadratic:
        x = x - alpha*g
        f, g = p.obj(x, gradient = True)
        nge += 1
        nfe += 1
    else:
        xtent = x - alpha*g
        f = p.obj(xtent, gradient = False)
        nfe += 1
        while(f > fref - 1e-4*alpha*(gnorm**2)):
            flag = 1
            if alpha < alpha_min:
                print('line search failed')
                raise Exception('line search failed')
#                 break
            alpha = alpha/2
            xtent = x - alpha*g
            f = p.obj(xtent, gradient = False) 
            nfe += 1
        x = xtent
        _, g = p.obj(x, gradient = True)
        nge += 1
    return x, f, g, alpha, nfe, nge, flag

# LMSD for general unconstrained optimization functions -----
def lmsd(p, M2 = 5, stepsize = 'bb1', option = 'fletcher', tol_lin_dep = 1e-8, tol = 1e-6, maxit = 5e4, alpha = 1, verbose = False, folder = 'data_out/'):
    
    name_algo = 'LMSD-' + option + '-' + stepsize
        
    if verbose:
        file_name = folder + p.name + '-' + name_algo + '-' + str(M2) + '.csv'
        df = open(file_name, 'wt') 
        writer = csv.writer(df)
        writer.writerow(('iters', 'f', 'gnorm', 'alpha', 'step_ind', 'step_len', 'flag'))
    
    time_start = time.time()
    
    # extract info from quadratic problem
    x = p.x0
    f, g = p.obj(x, gradient=True)  # objective and gradient
    
    # store initial f value 
    f0val = f
    
    # set tolerance
    gnorm = norm(g)
    tol = tol*gnorm #custom_norm(g)
    
    # init step
    steps, step_len, step_ind = [alpha], 1, 0
    
    # init memory
    G, mem = np.zeros((x.shape[0], M2)), np.zeros(M2)
    
    # counters for memory
    to_use, itmp = 0, 0
    
    # flag for tracking line search, set fref for line search
    flag, fref = 0, f
    
    # count iterations, number of fun eval, number of grad eval
    iters, nfe, nge, nsweep = 0, 1, 1, 0
    
    while iters < maxit:
        
        alpha = steps[step_ind]
        # set bounds on stepsize
        alpha = max(alpha_min, min(alpha, alpha_max)) 

        # info
        if verbose:
            writer.writerow((iters, f, gnorm, alpha, step_ind, step_len, flag))

        # update iterate
        g0, gnorm0, x0, f0 = g, gnorm, x, f
        x, f, g, alpha, nfe, nge, flag = line_search_lmsd(p, x0, alpha, g0, fref, gnorm0, nfe, nge, quadratic = False)
        
        # check convergence
        gnorm = norm(g)
#         if custom_norm(g) <= tol:
        if gnorm <= tol:
            break
            
#         if (flag == 1):
#             print('clear stack because we enter line search')
        
        if np.logical_and(flag == 0, gnorm > gnorm0):
            flag = 1
#             print('clear stack because nonmon gradient')
        
        # add vectors to memory
        G[:,itmp], mem[itmp] = g0, 1/alpha
        itmp = 0 if (itmp > M2-2) else itmp + 1
        
#         Daniela 18
#         to_use = step_ind + 1 if flag == 1 else min(M2, to_use + 1)
        to_use = min(M2, to_use + 1)
        
        # if stack is not empty
        if np.logical_and(step_ind < step_len-1, flag == 0): 
            step_ind += 1
        else:
#             print('we use', to_use, 'backgradients')
            
            nsweep += 1
            # order matrix of gradients
            jtmp = np.roll(list(range(0,M2)), -itmp)[-to_use:]
#             print(jtmp)
            
            # compute reduced hessian
            steps = reduced_hessian(G[:,jtmp], g, gnorm, mem[jtmp], option, stepsize, tol_lin_dep)
            step_len, step_ind = len(steps), 0
            
            # we keep as many backgradients as the number of stepsizes
            to_use = step_len 
            
#             print('we got', step_len, 'new stepsizes')
#             print(steps)
            
            # function value at the beginning of next sweep
            fref = f
            
        iters += 1
     
    return {'id_algo': name_algo, 'problem': p.name, 'size': p.n, 'memory':M2, 'nge':nge, 'nfe':nfe, 'f0': f0val, 'fstar': f, 'normg': gnorm, 'tol': tol, 'time_elapsed': time.time() - time_start, 'nsweep':nsweep}