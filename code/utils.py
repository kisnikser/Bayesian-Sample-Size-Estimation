import numpy as np

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
        
        #X_m = self.X[indexes, :] - это если np.array
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