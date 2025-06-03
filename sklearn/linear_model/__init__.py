class ElasticNet:
    def fit(self, X, y):
        self._y_mean = float(sum(y) / len(y)) if len(y) else 0.0
        return self
    def predict(self, X):
        return [self._y_mean for _ in range(len(X))]

class LinearRegression:
    def fit(self, X, y):
        self.intercept_ = float(sum(y) / len(y)) if len(y) else 0.0
        return self
    def predict(self, X):
        return [self.intercept_ for _ in range(len(X))]
