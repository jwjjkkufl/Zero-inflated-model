# Zero-inflated-model
## this is the 1st two-part model##
## the first part is for excessive zeros, following a binomial distribution
## the second part is for count data, following a poisson distribution 
## shortages: 
    ## only account for  count data, no continous data
    ## no correlate between two steps
    ## can be expanded to multipe parameters##

## achivenments: 
    ## customize MLE estimation 
    ## find out ZIP mass function

## confusion:
    ## "class" function in python
    ## "define" funcion within class

`from matplotlib import  pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

np.random.seed(123456789)
## two parameters needed to estimate
pi = 0.3
lambda_ = 2.

def zip_pmf(x, pi=pi, lambda_=lambda_): ## probability mass function##
    if pi < 0 or pi > 1 or lambda_ <= 0:
        return np.zeros_like(x) ##error information
    else:
        return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)
    
 ## First we generate 1,000 observations from the zero-inflated model.   
N = 1000

inflated_zero = stats.bernoulli.rvs(pi, size=N)
x = (1 - inflated_zero) * stats.poisson.rvs(lambda_, size=N)

plt.bar(x, align='center', alpha=0.5)
plt.show()

class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]

        return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            lambda_start = self.endog.mean()
            excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            start_params = np.array([excess_zeros, lambda_start])
            
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)
        
        
model = ZeroInflatedPoisson(x)
results = model.fit()       
print(results.summary())`

