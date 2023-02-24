import csv
import numpy as np
import scipy.linalg as la
import time
import warnings
warnings.filterwarnings('ignore')
alpha_min, alpha_max = 1e-30, 1e30

def iter_chol(GG, P):
    while True:
        try:
            R = la.cholesky(GG[np.ix_(P, P)])
            break
        except:
            P = np.delete(P, 0)
    return R, P

def augmented_GG(G, g):
    ll = G.shape[1]+1
    GG = np.zeros((ll,ll))
    GG[:-1,:-1] = G.T @ G
    GG[:-1,-1] = (G.T @ g).reshape(-1)
    GG[-1,:-1] = GG[:-1,-1]
    GG[-1,-1] = np.dot(g,g)
    return(GG)

def y_correction(R, alphas):
    sy = -np.diff(R[:,:-1].T.dot(R), axis = 1) / alphas 
    L = np.tril(-sy.T + sy)
    Z = (L.T * alphas.reshape(1,-1)) * alphas
    R = R[:,:-1]
    ZR = np.linalg.solve(R.T, Z)
    return np.linalg.solve(R.T, ZR.T).T

def wrap_npdiff(A, b):
    return -np.diff(A, axis = 1, append = b.reshape(-1,1))

def lyapunov(a, q):
    return la.solve_continuous_lyapunov(a, q)

def reduced_hessian(G, g, gnorm, mem, option, stepsize, tol_lin_dep):
    
    if option in ['fletcher', 'fletcher-pert', 'lya']:
        # compute Cholesky factor for [G g]
        P = np.array(range(G.shape[1]+1))
        GG = augmented_GG(G, g)

        # Cholesky factor
        R, P = iter_chol(GG, P)

        if P.shape[0] == 1:
            y = g - G[:,-1]
            if stepsize == 'bb1':
                B = np.array([[-mem[-1]*np.dot(G[:,-1], y)/np.dot(G[:,-1], G[:,-1])]])
            else:
                B = np.array([[-mem[-1]*np.dot(G[:,-1], y)/np.dot(y, y)]]) # already the BB2, not its reciprocal
        else:
            if option in ['fletcher', 'fletcher-pert']:
                AU = -np.diff(R, axis = 1) * mem[P[:-1]].reshape(1,-1) 
                AU = (np.linalg.solve(R[:-1,:-1].T, AU.T)).T # This gives upper Hessenberg!!
                # cf curtis and guo 2016 
                B, csi = AU[:-1,:], AU[-1,:]

                if option == 'fletcher-pert': # ONLY FOR BB1 ATM
                    B += y_correction(R[:-1,:], mem[P[:-1]].reshape(-1,1))
                    B = 0.5*(B + B.T)
                else:
                    B = np.tril(B,0) + np.tril(B,-1).T
                    
                if stepsize == 'bb2':
                    B = np.linalg.solve(B.T.dot(B) + np.outer(csi, csi), B)

            elif option == 'lya':
                sy = -np.diff(R[:-1,:-1].T.dot(R[:-1,:]), axis = 1) / mem[P[:-1]].reshape(-1,1) # divide row-wise
                s = R[:-1,:-1] / mem[P[:-1]].reshape(1,-1) if stepsize == 'bb1' else np.diff(R, axis = 1)
                B = lyapunov(s.T.dot(s), sy.T + sy)  
                B = 0.5*(B + B.T)
            
    elif option == 'lya-svd': # BB1 ONLY
        u, d, v = la.svd(-G / mem.reshape(1, -1), full_matrices=False) # SVD of S
        v = v.T

        # Check singular values and truncate
        P = d > tol_lin_dep*d[0]
        u, d, v = u[:,P], d[P], v[:,P]
        
        small_g = u.T.dot(g)
        sy = -wrap_npdiff(-(v.T * d.reshape(-1,1)) * mem.reshape(1,-1), small_g) * d.reshape(-1,1)
        sy = sy @ v
        ss = np.diag(d**2)
        B = lyapunov(ss, sy.T + sy)
        B = 0.5*(B + B.T)
        
    elif option == 'lya-qr': # BB1 ONLY
        Q, R, P = la.qr(G, pivoting=True, mode = 'economic')
        R1, P1 = R, np.argsort(P)

        # Check diagonal of R and truncate
        diagr = abs(np.diag(R))
        to_keep = diagr > tol_lin_dep*diagr[0]
        Q, R, P = Q[:,to_keep], R[np.ix_(to_keep, to_keep)], P[to_keep]

        small_g = G[:,P].T.dot(g)
        sy = wrap_npdiff(R.T @ R1[:len(P),P1], small_g)[:,P] / mem[P].reshape(-1,1)
        s = R / mem[P].reshape(1,-1)
        B = lyapunov(s.T.dot(s), sy.T + sy)
        B = 0.5*(B + B.T)
    
    # compute eigenvalues 
    w, _ = la.eig(B) 
    w = np.real(w)
    
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
    gnorm = np.linalg.norm(g)
    tol = tol*gnorm
    
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
        gnorm = np.linalg.norm(g)
        if gnorm <= tol:
            break
                    
        if np.logical_and(flag == 0, gnorm > gnorm0):
            flag = 1
        
        # add vectors to memory
        G[:,itmp], mem[itmp] = g0, 1/alpha
        itmp = 0 if (itmp > M2-2) else itmp + 1
        
#         Di Serafino 18
#         to_use = step_ind + 1 if flag == 1 else min(M2, to_use + 1)
        to_use = min(M2, to_use + 1)
        
        # if stack is not empty
        if np.logical_and(step_ind < step_len-1, flag == 0): 
            step_ind += 1
        else:
            nsweep += 1
            # order matrix of gradients
            jtmp = np.roll(list(range(0,M2)), -itmp)[-to_use:]
            
            # compute reduced hessian
            steps = reduced_hessian(G[:,jtmp], g, gnorm, mem[jtmp], option, stepsize, tol_lin_dep)
            step_len, step_ind = len(steps), 0
            
            # we keep as many backgradients as the number of stepsizes
            to_use = step_len 
            
            # function value at the beginning of next sweep
            fref = f
            
        iters += 1
     
    return {'id_algo': name_algo, 'problem': p.name, 'size': p.n, 'memory':M2, 'nge':nge, 'nfe':nfe, 'f0': f0val, 'fstar': f, 'normg': gnorm, 'tol': tol, 'time_elapsed': time.time() - time_start, 'nsweep':nsweep}