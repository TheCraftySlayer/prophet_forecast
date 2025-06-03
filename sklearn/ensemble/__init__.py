class GradientBoostingRegressor:
    def fit(self, X, y):
        # store mean of target as placeholder model
        self._y_mean = float(sum(y) / len(y)) if len(y) else 0.0
        return self
    def predict(self, X):
        return [self._y_mean for _ in range(len(X))]
