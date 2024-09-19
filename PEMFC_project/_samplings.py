import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def latin_hypercube_uniform(n,d):
    
    lower_limits = np.arange(0,n)/n
    upper_limits = np.arange(1,1+n)/n
    
    points = np.random.uniform(low=lower_limits, high=upper_limits, size=[d,n]).T
    if d==1: np.random.shuffle(points[0,None])
    else:
        for i in range(d):
            np.random.shuffle(points[i,:])
    
    return points


def KDE_samples(data, sample_size):
    
    def optimize_bandwidths(data):
        bandwidths = 10**np.linspace(-3,1,100)
        
        grid = GridSearchCV(KernelDensity(),
                    {'kernel':('gaussian', 'tophat'),
                    'bandwidth': bandwidths},
                   cv=5,n_jobs=-1)
        grid.fit(data[:,None])
        return grid.best_params_['kernel'],grid.best_params_['bandwidth']
    
    kernel,bw = optimize_bandwidths(data)
    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(data[:,None])
    new_samples = kde.sample(n_samples=sample_size)
    return new_samples.ravel()

def KDE_generate_params(get_sam, numbers):
    samples_1 = np.zeros((numbers,get_sam.shape[1]))
    for i in range(get_sam.shape[1]):
        param_new = KDE_samples(get_sam[:,i], numbers)
        param_new = np.abs(param_new)
        samples_1[:,i] = param_new
        
    return samples_1

def NaiveSMOTE(X, N=100, K=3):
    """
    {X}: minority class samples;
    {N}: Amount of SMOTE; default 100;
    {K} Number of nearest; default 5;
    """

    
    # {T}: Number of minority class samples; 
    T = X.shape[0]
    if N < 100:
        T = (N/100) * T
        N = 100
    N = (int)(N/100)
    
    numattrs = X.shape[1]
    samples = X[:T]
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(samples)

    Synthetic = np.zeros((T*N, numattrs))
    newindex = 0
    
    def Populate(N, i, nns, newindex):
        """
        Function to generate the synthetic samples.
        """
        for n in range(N):
            nn = np.random.randint(0, K)
            for attr in range(numattrs):
                dif = samples[nns[nn], attr] - samples[i, attr]
                gap = np.random.random()
                Synthetic[newindex, attr] = samples[i, attr] + gap*dif
            newindex += 1
        return newindex
    
    for i in range(T):
        nns = neigh.kneighbors([samples[i]], K, return_distance=False)
        newindex = Populate(N, i, nns[0], newindex)
    return Synthetic, newindex

def NaiveENN(x, L_c, y_c, K=5):
    """
        {x}: samples;
        {xL}: large samples;
        {y}: target classes
        {K} Number of nearest; default 5;
    """   
        
    def mode(data):
        values, counts = np.unique(data, return_counts=True)
        return values[np.where(counts == max(counts))[0][0]]
        
    c = np.unique(y_c)
    neigh = NearestNeighbors(n_neighbors=K, n_jobs=5)
    neigh.fit(x)
    dist, neighbors = neigh.kneighbors(x, n_neighbors=K, return_distance=True)
    
    xL_ind = np.where(y_c==L_c)[0].tolist()       
    xL = x[xL_ind,:]
    n = []
    for i in xL_ind :
        if not y_c[i] == mode(y_c[neighbors[i][1:]]):
            n.append(i)

    delindex = np.unique(n)
    new_y = np.delete(y_c,delindex)
    new_X = np.delete(x,delindex,axis=0)

    return new_X, new_y, delindex

def SMOTER(X,y, N=100, K=4):
    """
    {X}: minority class samples;
    {N}: Amount of SMOTE; default 100;
    {K} Number of nearest; default 5;
    """
    # {T}: Number of minority class samples; 
    T = X.shape[0]
    if N < 100:
        T = (N/100) * T
        N = 100
    N = (int)(N/100)
    
    numattrs = X.shape[1]
    samples_X = X[:T]
    samples_y = y[:T]
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(samples_X)
    Synthetic_X = np.zeros((T*N, numattrs))
    Synthetic_y = np.zeros((T*N))
    
    newindex = 0
    
    def Populate(N, i, nns, newindex):
        """
        Function to generate the synthetic samples.
        """
        for n in range(N):
            nn = np.random.randint(0, K)
            d1 = 0
            d2 = 0
            
            for attr in range(numattrs):
                dif = samples_X[nns[nn], attr] - samples_X[i, attr]
                gap = np.random.random()
                Synthetic_X[newindex, attr] = samples_X[i, attr] + gap*dif
                d1 = d1 + (Synthetic_X[newindex, attr] - samples_X[nns[nn], attr])**2
                d2 = d2 + (Synthetic_X[newindex, attr] - samples_X[i, attr])**2

                                           
            d1 = np.sqrt(d1)/(numattrs+1)
            d2 = np.sqrt(d2)/(numattrs+1)
            
            if d1+d2 == 0:
                Synthetic_y[newindex] = (samples_y[nns[nn]])
            else:
                Synthetic_y[newindex]=(d1*samples_y[nns[nn]]+d2*samples_y[i])/(d1+d2)
        
            newindex += 1
        return newindex
    
    for i in range(T):
        nns = neigh.kneighbors([samples_X[i]], K, return_distance=False)
        newindex = Populate(N, i, nns[0], newindex)
        
    return Synthetic_X, Synthetic_y
