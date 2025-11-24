import numpy as np

class GLR:
    def __init__(self,WindowSize,residual,mean,sigma):
        self.M = WindowSize #GLR Window Size
        self.N = len(residual)
        #0 Hypothesis mean and std
        self.mu_0 = mean
        self.sigma = sigma
        self.residual = residual
        self.g = np.zeros(self.N)             # Test statistics
        self.idx = np.ones(self.N, dtype=int) # Index of fault occurence sample estimation
        self.mu_1 = np.zeros(self.N)          # Estimated parameter

    def computeGlr(self):
        for k in range(self.M, self.N):       # For each new sample
            S =  np.zeros(self.M)        # Define the log-likelihood ratio
            z = self.residual[k-self.M+1 : k+1]     # Define the residual samples inside the window M
            for j in range (0,self.M):         # Iterate for all the time instants of the window
                sum_sq = 0
                for i in range(j,self.M):
                    sum_sq = sum_sq + (z[i] - self.mu_0)
                S[j] = sum_sq**2/((self.M - j)*2*self.sigma**2)
            self.g[k]   = np.max(S)
            self.idx[k] = np.argmax(S)
            #[g[k], idx[k]] = max(S); # Get the value of g(k) and the sample index
            self.mu_1[k] = np.sum(self.residual[k - self.M + self.idx[k] + 1 : k + 1] ) /(self.M - self.idx[k] + 1)
        return self.g, self.idx, self.mu_1