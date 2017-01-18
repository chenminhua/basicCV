from numpy import *
from scipy import linalg

class Camera(object):
    def __init__(self,P):
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center


    def project(self,X):
        x = dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]
        return x


    def factor(self):
        # factor first 3*3 part
        K,R = linalg.rq(self.P[:,:3])
        # make diagonal of K positive
        T = diag(sign(diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = dot(K,T)
        self.R = dot(T,R) # T is its own inverse
        self.t = dot(linalg.inv(self.K),self.P[:,3])

        return self.K, self.R, self.t


    def center(self):
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -dot(self.R.T,self.t)
            return self.c

def rotation_matrix(a):
    R = eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R


def rq(A):
    from scipy.linalg import qr

    Q,R = qr(flipud(A).T)
    R = flipud(R.T)
    Q = Q.T

    return R[:,::-1],Q[::-1,:]
