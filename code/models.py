import numpy as np
import scipy.stats as sps
from scipy.special import expit as expit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


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
        #self.w = np.linalg.inv(self.X.T @ self.X + self.alpha * np.identity(self.n)) @ self.X.T @ self.y
        self.w = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        #self.w = LinearRegression(fit_intercept=False).fit(self.X, self.y).coef_
        return self.w

    def predict(self, params, X=None):
        if X is None:
            X = self.X
        return X @ params

    def loglikelihood(self, params, X=None, y=None, sigma2=1):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        return -X.shape[0]/2*np.log(2*np.pi*sigma2) - 1/(2*sigma2)*np.sum((y - X @ params)**2)


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
    
    def sigmoid(self, x):
        x = np.asarray(x)
        return 1 / (1 + np.exp(-x))

    def loglikelihood(self, params, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        return np.sum(np.log(self.sigmoid(y * (X @ params))))