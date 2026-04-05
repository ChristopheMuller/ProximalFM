import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tabicl import TabICLRegressor

# =============================================================================
# Base Interface
# =============================================================================
class CATEEstimator:
    def __init__(self, add_W=True, add_Z=True):
        self.add_W = add_W
        self.add_Z = add_Z
    
    def fit(self, X, A, Y, W=None, Z=None):
        raise NotImplementedError
        
    def predict(self, X, W=None, Z=None):
        raise NotImplementedError
    
    def _concat_features(self, X, A=None, W=None, Z=None):
        """Utility to concatenate X with optional W, Z and optional A.

        - Respects the flags self.add_W and self.add_Z.
        - If A is provided it will be appended as a column (reshaped if needed).
        """
        features = [X]
        if self.add_W and (W is not None):
            features.append(W)
        if self.add_Z and (Z is not None):
            features.append(Z)
        if A is not None:
            # ensure column vector
            a = np.array(A)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            features.append(a)
        return np.hstack(features)

# =============================================================================
# Hyperclasses (Meta-Learners)
# =============================================================================
class S_Learner(CATEEstimator):
    """
    S-Learner: Stacks [X, W, Z, A] as features into a single model.
    CATE is estimated by predicting Y(A=1) - Y(A=0).
    """
    def __init__(self, base_model_cls, add_W=True, add_Z=True, **kwargs):
        # initialize flags in the common base
        super().__init__(add_W=add_W, add_Z=add_Z)
        self.model = base_model_cls(**kwargs)

    def fit(self, X, A, Y, W=None, Z=None):
        feat = self._concat_features(X, A=A, W=W, Z=Z)
        self.model.fit(feat, Y)
        return self

    def predict(self, X, W=None, Z=None):
        feat1 = self._concat_features(X, A=np.ones(X.shape[0]), W=W, Z=Z)
        feat0 = self._concat_features(X, A=np.zeros(X.shape[0]), W=W, Z=Z)
        return self.model.predict(feat1) - self.model.predict(feat0)


class T_Learner(CATEEstimator):
    """
    T-Learner: Fits two separate models for A=0 and A=1.
    """
    def __init__(self, base_model_cls, add_W=True, add_Z=True, **kwargs):
        super().__init__(add_W=add_W, add_Z=add_Z)
        self.model0 = base_model_cls(**kwargs)
        self.model1 = base_model_cls(**kwargs)

    def fit(self, X, A, Y, W=None, Z=None):
        feat = self._concat_features(X, A=None, W=W, Z=Z)
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
        feat = self._concat_features(X, A=None, W=W, Z=Z)
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