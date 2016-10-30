import numpy as np
import scipy.stats as sps
def winer_process_path(end_time, step, precision=10000):
    times = np.arange(0,end_time,step)
    c = step/np.pi
    ksi = np.random.normal(size=len(times))
    ksi[0] = 0
    values = (c * np.pi)**0.5 * np.cumsum(ksi)
    return times , values

class WinerProcess:
    def __init__(self,precision=10000):
        if precision == 0:
            precision = 1
        
        self.precision = precision
        self.eps = 1e-20
        self.ksi_0 = []
        self.ksi = dict()
        self.cumsum = []
        self._calc_X_vect = np.vectorize(self._calc_X)
        self._get_bound_vect = np.vectorize(self._get_bound)
    
#     returns k that time in [k*\pi; (k+1)*\pi]
    def _get_bound(self,time):
        return int(time/np.pi + self.eps)

    def _calc_X(self,t):
        k = self._get_bound(t)
        t -= np.pi * k
        i = np.arange(1,self.precision + 1)
        sin_all = np.sin(t * i)
        
        x = np.sum(self.ksi[k] * sin_all / i)
        x *= 2.0**0.5
        x += self.ksi_0[k] * t
        x /= (np.pi)**0.5
        if k > 0:
            x += (np.pi)**0.5 * self.cumsum[k-1]
        return x
    
    def _build_ksi(self,times):
        k = self._get_bound(np.max(times))
        if k >= len(self.ksi_0):
            self.ksi_0 = self.ksi_0 + list(np.random.normal(size=k - len(self.ksi_0) + 1))
            self.cumsum = np.cumsum(self.ksi_0)

        need_size = 0
#       build necessary ksi 
        k_unique = np.unique(self._get_bound_vect(times))
        need_size = sum(1 for k in k_unique if k not in self.ksi)
        new_ksi = np.random.normal(size=need_size*self.precision)
        #the second cyle
        for i in range(len(k_unique)):
            k = k_unique[i]
            if k not in self.ksi :
                self.ksi[k] = new_ksi[i*self.precision:(i+1)*self.precision]
    
    def __getitem__(self,times):
        self._build_ksi(times)
        return self._calc_X_vect(times) #the main cycle (legitimate)

