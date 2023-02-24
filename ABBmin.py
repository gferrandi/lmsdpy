import numpy as np
import time
import csv

alpha_min, alpha_max = 1e-30, 1e30

# line search
def line_search(p, x, alpha, g, fref, gnorm, nfe, nge, quadratic = False):
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

# compute stepsize    
def abb_min(g0, gnorm0, g, gnorm, alpha, mem):
    y = g - g0
    gy = np.dot(g0, y)
    if gy < 0:
        yy = np.linalg.norm(y)
        mem = np.append(mem, - alpha*gy/(yy**2))
        cosa = -gy/(gnorm0*yy)
        if cosa**2 < 0.8:
            alpha = min(mem)
        else:
            alpha = -alpha*(gnorm0**2)/gy 
        return alpha, mem[1:]
    else:
        return 1/max(1e-5, min(gnorm, 1)), mem

# gradient method for ABBmin, featuring nonmonotone line search for non quadratic functions        
def gradient_method(p, tol = 1e-6, maxit = 5e4, M = 10, M2 = 5, alpha = 1, quadratic = False, verbose = False, folder = 'data_out/'):
    
    name_algo = 'ABBmin'
    if verbose:
        file_name = folder + p.name + '-' + name_algo + '-' + str(M2) + '.csv'
        df = open(file_name, 'wt') 
        writer = csv.writer(df)
        writer.writerow(('iters', 'f', 'nfe', 'gnorm', 'alpha'))
    
    time_start = time.time()
    
    # extract info from pycutest
    x = p.x0
    f, g = p.obj(x, gradient=True)  # objective and gradient
    f0 = f
    gnorm = np.linalg.norm(g)
    
    fhist = np.ones(M)*f
    fref = max(fhist)
    mem = np.ones(M2)*np.inf
    nfe, nge, iters, tol = 1, 1, 0, tol*gnorm
    
    while iters < maxit:

        # info
        if verbose:
            writer.writerow((iters, f, nfe, gnorm, alpha))

        # line search
        g0, gnorm0 = g, gnorm
        x, f, g, alpha, nfe, nge, _ = line_search(p, x, alpha, g0, fref, gnorm0, nfe, nge, quadratic)
        
        if not quadratic:
            fhist = np.append(fhist[1:], f)
            fref = max(fhist)

        # check convergence
        gnorm = np.linalg.norm(g)
        if gnorm <= tol:
            break

        # update stepsize
        alpha, mem = abb_min(g0, gnorm0, g, gnorm, alpha, mem)
        
        # set bounds on stepsize
        if not quadratic:
            alpha = max(alpha_min, min(alpha, alpha_max))
        
        iters += 1
        
    return {'id_algo': name_algo, 'problem': p.name, 'size': p.n, 'memory':M2,
            'nge': nge, 'nfe': nfe, 'f0': f0, 'fstar': f, 'normg': gnorm, 'tol': tol, 'time_elapsed': time.time() - time_start}