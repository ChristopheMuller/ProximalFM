import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tabicl import TabICLRegressor

# =============================================================================
# Base Interface
# =============================================================================
class CATEEstimator:
    def fit(self, X, A, Y, W=None, Z=None):
        raise NotImplementedError
        
    def predict(self, X, W=None, Z=None):
        raise NotImplementedError

# =============================================================================
# Hyperclasses (Meta-Learners)
# =============================================================================
class S_Learner(CATEEstimator):
    """
    S-Learner: Stacks [X, W, Z, A] as features into a single model.
    CATE is estimated by predicting Y(A=1) - Y(A=0).
    """
    def __init__(self, base_model_cls, **kwargs):
        self.model = base_model_cls(**kwargs)

    def _prepare_features(self, X, A, W, Z):
        # Concatenate X, W, Z if provided, to adjust for proxies natively
        features = [X]
        if W is not None: features.append(W)
        if Z is not None: features.append(Z)
        features.append(A.reshape(-1, 1))
        return np.hstack(features)

    def fit(self, X, A, Y, W=None, Z=None):
        feat = self._prepare_features(X, A, W, Z)
        self.model.fit(feat, Y)
        return self

    def predict(self, X, W=None, Z=None):
        feat1 = self._prepare_features(X, np.ones(X.shape[0]), W, Z)
        feat0 = self._prepare_features(X, np.zeros(X.shape[0]), W, Z)
        return self.model.predict(feat1) - self.model.predict(feat0)


class T_Learner(CATEEstimator):
    """
    T-Learner: Fits two separate models for A=0 and A=1.
    """
    def __init__(self, base_model_cls, **kwargs):
        self.model0 = base_model_cls(**kwargs)
        self.model1 = base_model_cls(**kwargs)

    def _prepare_features(self, X, W, Z):
        features = [X]
        if W is not None: features.append(W)
        if Z is not None: features.append(Z)
        return np.hstack(features)

    def fit(self, X, A, Y, W=None, Z=None):
        feat = self._prepare_features(X, W, Z)
        mask0, mask1 = (A == 0), (A == 1)
        
        # Fit model 0 if there are control samples
        if mask0.sum() > 0:
            self.model0.fit(feat[mask0], Y[mask0])
        else:
            print("Warning: No control samples for T-Learner.")
            
        # Fit model 1 if there are treated samples
        if mask1.sum() > 0:
            self.model1.fit(feat[mask1], Y[mask1])
        else:
            print("Warning: No treated samples for T-Learner.")
            
        return self

    def predict(self, X, W=None, Z=None):
        feat = self._prepare_features(X, W, Z)
        return self.model1.predict(feat) - self.model0.predict(feat)

# =============================================================================
# Specific Subclasses
# =============================================================================
class S_OLS(S_Learner):
    def __init__(self):
        super().__init__(LinearRegression)

class T_OLS(T_Learner):
    def __init__(self):
        super().__init__(LinearRegression)

class S_RF(S_Learner):
    def __init__(self):
        super().__init__(RandomForestRegressor, n_estimators=100, max_depth=5, random_state=42)

class T_RF(T_Learner):
    def __init__(self):
        super().__init__(RandomForestRegressor, n_estimators=100, max_depth=5, random_state=42)

class S_TabICL(S_Learner):
    def __init__(self):
        super().__init__(TabICLRegressor)

class T_TabICL(T_Learner):
    def __init__(self):
        super().__init__(TabICLRegressor)

# =============================================================================
# Proximal 2SLS (Linear Baseline)
# =============================================================================
class P2SLS(CATEEstimator):
    def __init__(self):
        self.stage1 = LinearRegression()
        self.stage2 = LinearRegression()

    def fit(self, X, A, Y, W, Z):
        # Stage 1: Regress Outcome Proxy (W) on Treatment (A), Covariates (X), and Treatment Proxy (Z)
        s1_feat = np.hstack([A.reshape(-1, 1), X, Z])
        self.stage1.fit(s1_feat, W)
        W_hat = self.stage1.predict(s1_feat)

        # Stage 2: Regress Outcome (Y) on Treatment (A), Covariates (X), and predicted Proxy (W_hat)
        s2_feat = np.hstack([A.reshape(-1, 1), X, W_hat])
        self.stage2.fit(s2_feat, Y)
        
        # The causal effect of A is the first coefficient
        self.coef_A = self.stage2.coef_[0]
        return self

    def predict(self, X, W=None, Z=None):
        # P2SLS yields a constant ATE (assumes homogeneous effect in this linear form)
        return np.full(X.shape[0], self.coef_A)