import numpy as np


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

#Inf norm
def inorm(x):
    return np.linalg.norm(x,np.inf)

#Froebenius norm
def fnorm(x):
    return np.linalg.norm(x)

def nearcor(R, eig_tol = 1.0e-6, conv_tol = 1.0e-7, posd_tol = 1.0e-8, maxits = 100, verbose = False):
    
    
    if (not (type(R) is np.ndarray and check_symmetric(R))):
        print("Error: Input matrix R must be square and symmetric")
        return

    n = R.shape[1]
    U = np.zeros((n, n))
    Y = R.copy()
    it = 0

    while True:
        T = Y - U
        e = np.linalg.eig(Y)
        Q = e[1]
        d = e[0]
        D = np.diag(d)
        
        p = (d>eig_tol*d[0])
        X = np.dot(Q.T[p].T,  np.dot((D[p].T)[p].T, Q.T[p]))
        U = X - T
        X = (X + X.T)/2.0
        np.fill_diagonal(X,1.0)
        conv = inorm(Y-X) / inorm(Y)
        it += 1
        
        if verbose:
            print("iter = {}, conv= {}".format(it,conv))

        if conv <= conv_tol:
            converged = True
            break
        elif it == maxits:
            print("nearcor did not converge in {} iterations".format(it))
            converged = False
            break
        Y = X.copy()

    X = (X + X.T)/2.0
    e = np.linalg.eig(X)
    d = e[0]
    Eps = posd_tol * abs(d[0])
    
    if (d[n-1] < Eps):
        d[d < Eps] = Eps
        Q = e[1]
        o_diag = np.diag(X)
        print(o_diag)
        X = np.dot(Q, (d * Q).T)
        print(X)
        D = np.sqrt(np.maximum(Eps, o_diag)/np.diag(X))
        print(D)
        X = (D * X.T).T * D
        print(X)
        X = (X + X.T)/2.0
        print(X)
    
    np.fill_diagonal(X,1.0)

    return (X, fnorm(R-X), it, converged)
