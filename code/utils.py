import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm
myparams = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'font.family': 'Djvu Serif',
    'font.size': 14,
    'axes.grid': True,
    'grid.alpha': 0.1,
    'lines.linewidth': 2
}
plt.rcParams.update(myparams)
from models import RegressionModel, LogisticModel


class Dataset:

    def __init__(self, X, y, task='regression'):
        """
        Constructor method
        """
        self.X = X
        self.y = y
        self.task = task

        if task == 'classification':
            self.labels = np.unique(self.y)

        self.m = self.y.shape[0]
        self.n = self.X.shape[1]

    def sample(self, m=None, duplications=True):
        """
        Parameters
        ----------
        m: int
            Subset sample size, must be greater than number of feature
        duplications: bool
        """
        if m is None:
            m = self.m

        if m <= self.n:
            raise ValueError(
                "The m={} value must be greater than number of feature={}".format(m, self.n))
        
        if self.task == 'classification' and m <= len(self.labels):
            raise ValueError(
                "The m={} value must be greater than number of classes={}".format(m, len(self.labels)))
        
        if duplications:
            indexes = np.random.randint(low = 0, high=self.m, size=m)
        else:
            indexes = np.random.permutation(self.m)[:m]
        
        
        if isinstance(self.X, np.ndarray):
            X_m = self.X[indexes, :] # - это если np.array
        else:
            X_m = self.X.loc[indexes]
        y_m = self.y[indexes]

        if self.task == 'classification':
            while True:
                if isinstance(self.X, np.ndarray):
                    X_m = self.X[indexes, :] # - это если np.array
                else:
                    X_m = self.X.loc[indexes]
                y_m = self.y[indexes]
                if len(np.unique(y_m)) < len(self.labels):
                    indexes = np.random.randint(low=0, high=self.m, size=m)
                else:
                    break

        return X_m, y_m

    def train_test_split(self, test_size = 0.3, safe=True):

        X = self.X
        y = self.y

        M = int(self.m * test_size)

        indexes_test = np.random.permutation(self.m)[:M]
        indexes_train = np.random.permutation(self.m)[M:]

        X_train = X[indexes_train, :]
        X_test = X[indexes_test, :]
        y_train = y[indexes_train]
        y_test = y[indexes_test]

        if safe:
            while ((y_train == 0).all() or (y_train == 1).all() or (y_test == 0).all() or (y_test == 1).all()):
                indexes_test = np.random.permutation(self.m)[:M]
                indexes_train = np.random.permutation(self.m)[M:]
                X_train = X[indexes_train, :]
                X_test = X[indexes_test, :]
                y_train = y[indexes_train]
                y_test = y[indexes_test]

        return X_train, X_test, y_train, y_test
    
    
def D(means, variances):
    return variances

def M(means, variances):
    return np.abs(np.diff(means, n=1))

def func_mean_approx(k, w):
    return w[0] + w[1] * np.exp(w[2] * k)

def func_var_approx(k, w):
    return w[0] + w[1] * np.exp(w[2] * k)
    

class Estimator:
    def __init__(self, sample_sizes, means=None, variances=None, divergences=None, scores=None):
        self.sample_sizes = sample_sizes
        self.means = means # for rate definition
        self.variances = variances # for variance definition
        self.divergences = divergences # for kl-div definition
        self.scores = scores # for s-score definiton
        self.eps = {}
        self.m_star = {}
    
    def sufficient_sample_size(self, eps=1e-4, method="variance"):
    
        if method not in ["variance", "rate", "kl-div", "s-score"]:
            raise NotImplementedError
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        divergences = self.divergences
        scores = self.scores
    
        m_star = np.inf
        
        if method == "variance":
            for k, var in zip(sample_sizes, D(means, variances)):
                if var <= eps and m_star == np.inf:
                    m_star = k
                elif var > eps:
                    m_star = np.inf
                    
        elif method == "rate":
            for k, diff in zip(sample_sizes[:-1], M(means, variances)):
                if diff <= eps and m_star == np.inf:
                    m_star = k
                elif diff > eps:
                    m_star = np.inf
            
        elif method == "kl-div":
            for k, div in zip(sample_sizes, divergences):
                if div <= eps and m_star == np.inf:
                    m_star = k
                elif div > eps:
                    m_star = np.inf
            
        elif method == "s-score":
            for k, score in zip(sample_sizes, scores):
                if score >= 1 - eps and m_star == np.inf:
                    m_star = k
                elif score < 1 - eps:
                    m_star = np.inf
            
        self.eps[method] = eps
        self.m_star[method] = m_star

class Forecaster:
    
    def __init__(self, sample_sizes, means, variances) -> None:
        self.sample_sizes = sample_sizes
        self.means = means
        self.variances = variances
        
    
    def approx(self, func_mean=func_mean_approx, func_var=func_var_approx, n_means=3, n_variances=3, w0_means=None, w0_variances=None, train_size=0.5, verbose=False):
        
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        self.train_size = train_size
        
        # initial point for optimizing parameters w
        #w0_means = np.random.normal(size=n_means) if w0_means is None else w0_means
        #w0_variances = np.random.normal(size=n_variances) if w0_variances is None else w0_variances
        w0_means = np.zeros(n_means) if w0_means is None else w0_means
        w0_variances = np.zeros(n_variances) if w0_variances is None else w0_variances
        # number of points in train sample
        M = int(train_size*sample_sizes.size)
            
        X_train_means = sample_sizes[:M]
        y_train_means = means[:M]
        
        X_train_variances = sample_sizes[:M]
        y_train_variances = variances[:M]
            
        # find parameters w, that minimize MSE between log-likelihood (-loss) mean and it's approximation
        # start optimizing from w = w0
        means_minimum = minimize(lambda w: ((func_mean(X_train_means, w) - y_train_means)**2).mean(), w0_means)
        w_means = means_minimum.x
        variances_minimum = minimize(lambda w: ((func_var(X_train_variances, w) - y_train_variances)**2).mean(), w0_variances)
        w_variances = variances_minimum.x
        
        means_approximation = func_mean(sample_sizes, w_means)
        variances_approximation = func_var(sample_sizes, w_variances)

        self.means_approximation = means_approximation
        self.variances_approximation = variances_approximation


class Visualizer:
    
    def __init__(self, sample_sizes, means, variances, divergences=None, scores=None, loss=False, format="pdf") -> None:
        self.sample_sizes = sample_sizes
        self.means = means
        self.variances = variances
        self.divergences = divergences
        self.scores = scores
        self.loss = loss
        self.format = format
        self.color_between = "gray" if self.format == "eps" else None
    
    def plot_bootstrap(self, save=False, filename=None):
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        loss = self.loss
    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        stds = np.sqrt(variances)

        ax1.plot(sample_sizes, means, label=r"$\mathbb{E}_{\hat{\mathbf{w}}_{k}} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.fill_between(sample_sizes, means - stds, means + stds, alpha=0.3, color=self.color_between, label=r"$\pm \sqrt{\mathbb{D}_{\hat{\mathbf{w}}_{k}} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)}$")
        ax1.set_xlabel(r"$k$")
        if loss:
            ax1.set_ylabel(r"$-Loss$")
        else:
            ax1.set_ylabel(r"$l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.legend(loc="lower right")

        ax2.plot(sample_sizes, D(means, variances))
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$D(k)$")
        ax2.set_yscale('log')

        ax3.plot(sample_sizes[:-1], M(means, variances))
        ax3.set_xlabel(r"$k$")
        ax3.set_ylabel(r"$M(k)$")
        ax3.set_yscale('log')

        fig.tight_layout()
        if save:
            plt.savefig(filename+f".{self.format}", bbox_inches="tight")
        plt.show()

    
    def plot_bootstrap_sufficient(self, estimator: Estimator, save=False, filename=None):
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        loss = self.loss
    
        eps_variance = estimator.eps["variance"]
        m_star_variance = estimator.m_star["variance"]
        eps_rate = estimator.eps["rate"]
        m_star_rate = estimator.m_star["rate"]
        
        m_star_variance_idx = sample_sizes.tolist().index(m_star_variance)
        m_star_rate_idx = sample_sizes.tolist().index(m_star_rate)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        stds = np.sqrt(variances)

        ax1.plot(sample_sizes, means, label=r"$\mathbb{E}_{\hat{\mathbf{w}}_{k}} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.fill_between(sample_sizes, means - stds, means + stds, alpha=0.3, color=self.color_between, label=r"$\pm \sqrt{\mathbb{D}_{\hat{\mathbf{w}}_{k}} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)}$")
        ax1.vlines(m_star_variance, min(means - stds), means[m_star_variance_idx], linestyle='dashed', linewidth=1)
        ax1.scatter(m_star_variance, means[m_star_variance_idx], marker='o')
        ax1.vlines(m_star_rate, min(means - stds), means[m_star_rate_idx], linestyle='dashed', linewidth=1)
        ax1.scatter(m_star_rate, means[m_star_rate_idx], marker='^')
        ax1.set_xlabel(r"$k$")
        if loss:
            ax1.set_ylabel(r"$-Loss$")
        else:
            ax1.set_ylabel(r"$l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.legend(loc="lower right")

        ax2.plot(sample_sizes, D(means, variances))
        ax2.vlines(m_star_variance, 0, D(means, variances)[m_star_variance_idx], linestyle='dashed', linewidth=1)
        ax2.scatter(m_star_variance, D(means, variances)[m_star_variance_idx], marker='o', label="D-sufficient")
        ax2.vlines(m_star_rate, 0, D(means, variances)[m_star_rate_idx], linestyle='dashed', linewidth=1)
        ax2.scatter(m_star_rate, D(means, variances)[m_star_rate_idx], marker='^')
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$D(k)$")
        ax2.set_yscale('log')
        ax2.legend(loc="upper right")

        ax3.plot(sample_sizes[:-1], M(means, stds))
        ax3.vlines(m_star_variance, 0, M(means, stds)[m_star_variance_idx], linestyle='dashed', linewidth=1)
        ax3.scatter(m_star_variance, M(means, stds)[m_star_variance_idx], marker='o')
        ax3.vlines(m_star_rate, 0, M(means, stds)[m_star_rate_idx], linestyle='dashed', linewidth=1)
        ax3.scatter(m_star_rate, M(means, stds)[m_star_rate_idx], marker='^', label="M-sufficient")
        ax3.set_xlabel(r"$k$")
        ax3.set_ylabel(r"$M(k)$")
        ax3.set_yscale('log')
        ax3.legend(loc="upper right")

        plt.tight_layout()
        if save:
            plt.savefig(filename+f".{self.format}", bbox_inches="tight")
        plt.show()
        
    def plot_posterior_sufficient(self, estimator: Estimator, save=False, filename=None):
        sample_sizes = self.sample_sizes
        divergences = self.divergences
        scores = self.scores
        
        eps_div = estimator.eps["kl-div"]
        m_star_div = estimator.m_star["kl-div"]
        eps_score = estimator.eps["s-score"]
        m_star_score = estimator.m_star["s-score"]

        m_star_div_idx = sample_sizes.tolist().index(m_star_div)
        m_star_score_idx = sample_sizes.tolist().index(m_star_score)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(sample_sizes, divergences, zorder=1)
        ax1.vlines(m_star_div, 0, divergences[m_star_div_idx], 
                   linestyle='dashed', linewidth=1, zorder=2, label="KL-sufficient")
        ax1.scatter(m_star_div, divergences[m_star_div_idx], marker='o', zorder=2)
        ax1.vlines(m_star_score, 0, divergences[m_star_score_idx], 
                   linestyle='dotted', linewidth=1, zorder=2)
        ax1.scatter(m_star_score, divergences[m_star_score_idx], marker='^', zorder=2)
        ax1.set_xlabel(r"$k$")
        ax1.set_ylabel(r"$KL(k)$")
        ax1.set_yscale('log')
        ax1.legend(loc="upper right")

        ax2.plot(sample_sizes, scores, zorder=1)
        ax2.vlines(m_star_div, 0, scores[m_star_div_idx], 
                   linestyle='dashed', linewidth=1, zorder=2)
        ax2.scatter(m_star_div, scores[m_star_div_idx], marker='o', zorder=2)
        ax2.vlines(m_star_score, 0, scores[m_star_score_idx], 
                   linestyle='dotted', linewidth=1, zorder=2, label="S-sufficient")
        ax2.scatter(m_star_score, scores[m_star_score_idx], marker='^', zorder=2)
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$S(k)$")
        #ax2.set_yscale('log')
        ax2.set_ylim(min(scores), 1)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        if save:
            plt.savefig(filename+f".{self.format}", bbox_inches="tight")
        plt.show()
        
    def plot_bootstrap_approximation(self, forecaster: Forecaster, save=False, filename=None):
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        loss = self.loss

        means_approximation = forecaster.means_approximation
        variances_approximation = forecaster.variances_approximation

        train_size = forecaster.train_size
        train_bound = int(train_size * sample_sizes.size)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(sample_sizes, means, label='actual')
        ax1.plot(sample_sizes, means_approximation, label='approximation')
        ax1.vlines(sample_sizes[train_bound], min(means), max(means), colors='red', linestyle='dashed', linewidth=1, label='train/test')
        ax1.legend()
        ax1.set_xlabel(r"$k$")
        if loss:
            ax1.set_ylabel(r"$-Loss$")
        else:
            ax1.set_ylabel(r"$\mathbb{E}_{\hat{\mathbf{w}}_{k}} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")

        ax2.plot(sample_sizes, variances, label='actual')
        ax2.plot(sample_sizes, variances_approximation, label='approximation')
        ax2.vlines(sample_sizes[train_bound], min(variances), max(variances), colors='red', linestyle='dashed', linewidth=1, label='train/test')
        ax2.legend()
        ax2.set_xlabel(r"$k$")
        if loss:
            ax2.set_ylabel("Variance of loss")
        else:
            ax2.set_ylabel(r"$\mathbb{D}_{\hat{\mathbf{w}}_{k}} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax2.set_yscale('log')

        plt.tight_layout()
        if save:
            plt.savefig(filename+f".{self.format}", bbox_inches="tight")
        plt.show()
        
        
    def plot_bootstrap_sufficient_approximation(self, estimator: Estimator, forecaster: Forecaster, save=False, filename=None):
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        loss = self.loss
    
        eps_variance = estimator.eps["variance"]
        m_star_variance = estimator.m_star["variance"]
        eps_rate = estimator.eps["rate"]
        m_star_rate = estimator.m_star["rate"]
        
        m_star_variance_idx = sample_sizes.tolist().index(m_star_variance)
        m_star_rate_idx = sample_sizes.tolist().index(m_star_rate)
        
        means_approximation = forecaster.means_approximation
        variances_approximation = forecaster.variances_approximation

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        stds = np.sqrt(variances)
        stds_approximation = np.sqrt(variances_approximation)

        ax1.plot(sample_sizes, means, label="True")
        ax1.fill_between(sample_sizes, means - stds, means + stds, color=self.color_between, alpha=0.3)
        ax1.plot(sample_sizes, means_approximation, label="Approximation")
        ax1.fill_between(sample_sizes, means_approximation - stds_approximation, means_approximation + stds_approximation, color=self.color_between,  alpha=0.3)
        ax1.vlines(m_star_variance, min(means_approximation - stds_approximation), means_approximation[m_star_variance_idx], linestyle='dashed', linewidth=1)
        ax1.scatter(m_star_variance, means_approximation[m_star_variance_idx], marker='o')
        ax1.vlines(m_star_rate, min(means_approximation - stds_approximation), means_approximation[m_star_rate_idx], linestyle='dashed', linewidth=1)
        ax1.scatter(m_star_rate, means_approximation[m_star_rate_idx], marker='^')
        ax1.set_xlabel(r"$k$")
        if loss:
            ax1.set_ylabel(r"$-Loss$")
        else:
            ax1.set_ylabel(r"$l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.legend(loc="lower right")

        ax2.plot(sample_sizes, D(means, variances), label="True")
        ax2.plot(sample_sizes, D(means_approximation, variances_approximation), label="Approximation")
        ax2.vlines(m_star_variance, 0, D(means_approximation, variances_approximation)[m_star_variance_idx], linestyle='dashed', linewidth=1)
        ax2.scatter(m_star_variance, D(means_approximation, variances_approximation)[m_star_variance_idx], marker='o', label=f"D-sufficient")
        ax2.vlines(m_star_rate, 0, D(means_approximation, variances_approximation)[m_star_rate_idx], linestyle='dashed', linewidth=1)
        ax2.scatter(m_star_rate, D(means_approximation, variances_approximation)[m_star_rate_idx], marker='^')
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$D(k)$")
        ax2.set_yscale('log')
        ax2.legend(loc="upper right")

        ax3.plot(sample_sizes[:-1], M(means, variances), label="True")
        ax3.plot(sample_sizes[:-1], M(means_approximation, variances_approximation), label="Approximation")
        ax3.vlines(m_star_variance, 0, M(means_approximation, variances_approximation)[m_star_variance_idx], linestyle='dashed', linewidth=1)
        ax3.scatter(m_star_variance, M(means_approximation, variances_approximation)[m_star_variance_idx], marker='o')
        ax3.vlines(m_star_rate, 0, M(means_approximation, variances_approximation)[m_star_rate_idx], linestyle='dashed', linewidth=1)
        ax3.scatter(m_star_rate, M(means_approximation, variances_approximation)[m_star_rate_idx], marker='^', label=f"M-sufficient")
        ax3.set_xlabel(r"$k$")
        ax3.set_ylabel(r"$M(k)$")
        ax3.set_yscale('log')
        ax3.legend(loc="upper right")

        plt.tight_layout()
        if save:
            plt.savefig(filename+f".{self.format}", bbox_inches="tight")
        plt.show()
        
        
def get_norms_and_eigvals(X, B=100):
    """
    Args:
        X: object-feature matrix
        B: number of epochs
        
    Returns:
        norms: || Xkp1.T @ Xkp1 - Xk.T @ Xk || 
        eigvals: lambda(Xk.T @ Xk)
    """
    norms = []
    eigvals = []

    for _ in tqdm(range(B)):
        tmp = []
        tmp_eigvals = []
        k = X.shape[0] - 1
        Xkp1 = X
        while k > X.shape[1]:
            index = np.random.randint(k)
            Xk = np.delete(Xkp1, index, axis=0)
            tmp.append(np.linalg.norm(Xkp1.T @ Xkp1 - Xk.T @ Xk))
            tmp_eigvals.append(np.linalg.eigvalsh(Xk.T @ Xk)[0])
            Xkp1 = Xk
            k -= 1
        norms.append(tmp)
        eigvals.append(tmp_eigvals)

    norms = np.array(norms)
    norms = norms.mean(axis=0)[::-1]
    eigvals = np.array(eigvals)
    eigvals = eigvals.mean(axis=0)[::-1]
    
    return norms, eigvals


def get_means_and_variances(X, y, sample_sizes=None, task="regression", sigma2=1, B=100):
    if sample_sizes is None:
        sample_sizes = np.arange(X.shape[1]+1, X.shape[0])
        
    means = []
    variances = []
    
    dataset = Dataset(X, y, task)
    Model = RegressionModel if task == "regression" else LogisticModel

    for k in tqdm(sample_sizes):
        tmp = []
        for _ in range(B):
            X_k, y_k = dataset.sample(k)
            model = Model(X_k, y_k)
            w_hat = model.fit()
            if task == "regression":  
                tmp.append(model.loglikelihood(w_hat, X, y, sigma2))
            else:
                tmp.append(model.loglikelihood(w_hat, X, y))
        tmp = np.array(tmp)
        means.append(tmp.mean())
        variances.append(tmp.var())
        
    means = np.array(means)
    variances = np.array(variances)
    
    return means, variances


def posterior_parameters(m0, Sigma0, X, y, sigma2=1):
    Sigma = np.linalg.inv(np.linalg.inv(Sigma0) + 1/sigma2 * X.T @ X)
    m = Sigma @ (1/sigma2 * X.T @ y + np.linalg.inv(Sigma0) @ m0)
    return m, Sigma


def KL(mk, Sk, mkp1, Skp1):
    return 1/2 * (np.trace(np.linalg.inv(Skp1) @ Sk) + (mkp1 - mk) @ np.linalg.inv(Skp1) @ (mkp1 - mk) - mk.size + np.log(np.linalg.det(Skp1) / np.linalg.det(Sk)))


def s_score(mk, Sk, mkp1, Skp1):
    return np.exp(-1/2 * ((mkp1 - mk) @ np.linalg.inv(Sk + Skp1) @ (mkp1 - mk)))


def get_divergences_and_scores(X, y, m0, Sigma0, B=100):
    divergences = []
    scores = []

    for _ in tqdm(range(B)):
        
        tmp_divergences = []
        tmp_scores = []
        k = X.shape[0] - 1
        Xkp1, ykp1 = X, y
        mkp1, Skp1 = posterior_parameters(m0, Sigma0, Xkp1, ykp1)

        while k >= X.shape[1] + 1:
            index = np.random.randint(k)
            Xk, yk = np.delete(Xkp1, index, axis=0), np.delete(ykp1, index, axis=0)
            mk, Sk = posterior_parameters(m0, Sigma0, Xk, yk)
            tmp_divergences.append(KL(mk, Sk, mkp1, Skp1))
            tmp_scores.append(s_score(mk, Sk, mkp1, Skp1))
            Xkp1, ykp1 = Xk, yk
            mkp1, Skp1 = mk, Sk
            k -= 1
            
        divergences.append(tmp_divergences)
        scores.append(tmp_scores)
        
    divergences = np.mean(divergences, axis=0)[::-1]
    scores = np.mean(scores, axis=0)[::-1]
    
    return divergences, scores