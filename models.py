import numpy as np
import scipy.stats as sps
from scipy.special import expit as expit
from sklearn.linear_model import LogisticRegression


class LinearModel:
    def __init__(self, X, y, **kwargs):
        pass
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self, params, X=None):
        raise NotImplementedError
    
    def loglikelihood(self, params):
        raise NotImplementedError

    def score(self, params):
        raise NotImplementedError

    def hessian(self, params):
        raise NotImplementedError

    def loglikelihood_fixed(self, params):
        raise NotImplementedError

    def score_fixed(self, params):
        raise NotImplementedError

    def hessian_fixed(self, params):
        raise NotImplementedError

    def covariance(self, params):
        raise NotImplementedError

class RegressionModel(LinearModel):
    """
    Description for linear regresion model
    """
    def __init__(self, X, y, **kwargs):
        """
        Constructor method.
        """
        self.X = X
        self.y = y
        self.alpha = kwargs.pop('alpha', 0.01)
        self.w = None

        self.m = self.y.shape[0]
        self.n = self.X.shape[1]

        self.prior = sps.multivariate_normal(
            mean=np.zeros(self.n), 
            cov=self.alpha**(-1) * np.identity(self.n)
        )

    def fit(self):
        self.w = np.linalg.inv(self.X.T @ self.X + self.alpha * np.identity(self.n)) @ self.X.T @ self.y
        return self.w

    def predict(self, params, X=None):
        if X is None:
            X = self.X
        return X @ params

    def loglikelihood(self, params):
        # There we assume that y ~ N(wx,1)
        # And we don't consider last term with (2 * pi * sigma^2)^(-m/2)
        return -0.5 * np.sum((self.y - self.X @ params)**2)
    
    def loglikelihood_fixed(self, params):
        return self.loglikelihood(params) + self.prior.logpdf(params)

    def score(self, params):
        # score = d/dw log-likelihood(w) (Wikipedia)
        return self.X.T @ self.y - self.X.T @ self.X @ params

    def score_fixed(self, params):
        # score = d/dw log-likelihood(w) (Wikipedia)
        return self.score(params) - self.alpha * params

    def hessian(self, params):
        # hessian = d^2/dw^2 log-likelihood(w)
        return -self.X.T @ self.X

    def hessian_fixed(self, params):
        # hessian = d^2/dw^2 log-likelihood(w)
        return self.hessian(params) - self.alpha*np.identity(self.n)
    
    def covariance(self, params):
        # Dw = 1 / I(w), where I(w) is Fisher information matrix, i.e.
        # I(w) = -d^2/dw^2 log-likelihood(w) = -hessian(w)
        return np.linalg.inv(-self.hessian_fixed(params))


class LogisticModel(LinearModel):
    """
    Description for linear logistic model
    """
    def __init__(self, X, y, **kwargs):
        """
        Constructor method.
        """
        self.X = X
        self.y = y
        self.alpha = kwargs.pop('alpha', 0.01)
        self.w = None

        self.m = y.shape[0]
        self.n = X.shape[1]

        self.prior = sps.multivariate_normal(
            mean = np.zeros(self.n), 
            cov = self.alpha**(-1) * np.identity(self.n) # здесь была ошибка у Андрея, alpha, а не alpha^-1
        )

    def fit(self):
        model_sk_learn = LogisticRegression(C = 1./self.alpha)
        model_sk_learn.fit(self.X, self.y)
        self.w = model_sk_learn.coef_[0]
        return self.w

    def predict(self, params, X=None):
        if X is None:
            X = self.X
        return expit(X @ params)

    def loglikelihood(self, params):
        epsilon = 1e-10
        q = 2 * self.y - 1 # labels 0, 1 -> -1, 1
        res = expit(q * np.dot(self.X, params))
        res = res + (res < epsilon)*epsilon
        return np.sum(np.log(res))
    
    def loglikelihood_fixed(self, params):
        return self.loglikelihood(params) + self.prior.logpdf(params)

    def score(self, params):
        # score = d/dw log-likelihood(w) (Wikipedia)
        # there we get score for labels 0, 1
        theta = expit(self.X @ params)
        return np.dot(self.y - theta, self.X)
    
    def score_fixed(self, params):
        return self.score(params) - self.alpha*params

    def hessian(self, params):
        theta = expit(self.X @ params)
        return -np.dot(theta * (1-theta) * self.X.T, self.X)

    def hessian_fixed(self, params):
        return self.hessian(params) - self.alpha*np.identity(self.n)

    def covariance(self, params):
        return np.linalg.inv(-self.hessian_fixed(params))