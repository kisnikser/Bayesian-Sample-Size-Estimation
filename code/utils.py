import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
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


class Dataset(object):

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
                #X_m = self.X[indexes, :]
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
    
    def __init__(self, sample_sizes, means, variances) -> None:
        self.sample_sizes = sample_sizes
        self.means = means
        self.variances = variances
        self.eps = {}
        self.m_star = {}
    
    
    def sufficient_sample_size(self, eps=1e-4, method="variance"):
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
    
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
    
    def __init__(self, sample_sizes, means, variances, loss=False, format="pdf") -> None:
        self.sample_sizes = sample_sizes
        self.means = means
        self.variances = variances
        self.loss = loss
        self.format = format
    
    def plot_bootstrap(self, save=False, filename=None):
    
        sample_sizes = self.sample_sizes
        means = self.means
        variances = self.variances
        loss = self.loss
    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        stds = np.sqrt(variances)

        ax1.plot(sample_sizes, means, label=r"$\mathbb{E}_{\mathfrak{D}_k} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.fill_between(sample_sizes, means - stds, means + stds, alpha=0.3, label=r"$\pm \sqrt{\mathbb{D}_{\mathfrak{D}_k} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)}$")
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
            plt.savefig(filename, bbox_inches="tight", format=self.format)
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

        ax1.plot(sample_sizes, means, label=r"$\mathbb{E}_{\mathfrak{D}_k} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax1.fill_between(sample_sizes, means - stds, means + stds, alpha=0.3, label=r"$\pm \sqrt{\mathbb{D}_{\mathfrak{D}_k} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)}$")
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
            plt.savefig(filename, bbox_inches="tight", format=self.format)
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
            ax1.set_ylabel(r"$\mathbb{E}_{\mathfrak{D}_k} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")

        ax2.plot(sample_sizes, variances, label='actual')
        ax2.plot(sample_sizes, variances_approximation, label='approximation')
        ax2.vlines(sample_sizes[train_bound], min(variances), max(variances), colors='red', linestyle='dashed', linewidth=1, label='train/test')
        ax2.legend()
        ax2.set_xlabel(r"$k$")
        if loss:
            ax2.set_ylabel("Variance of loss")
        else:
            ax2.set_ylabel(r"$\mathbb{D}_{\mathfrak{D}_k} l(\mathfrak{D}_m, \hat{\mathbf{w}}_k)$")
        ax2.set_yscale('log')

        plt.tight_layout()
        if save:
            plt.savefig(filename, bbox_inches="tight", format=self.format)
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
        ax1.fill_between(sample_sizes, means - stds, means + stds, alpha=0.3)
        ax1.plot(sample_sizes, means_approximation, label="Approximation")
        ax1.fill_between(sample_sizes, means_approximation - stds_approximation, means_approximation + stds_approximation, alpha=0.3)
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
            plt.savefig(filename, bbox_inches="tight", format=self.format)
        plt.show()