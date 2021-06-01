import cvxpy as cp
import numpy as np
import Kernel

class L2IterativeBoost:
    """
    """
    def __init__(self, lambda_val=0.1, nu_val=2, sigma_sq_val=1):
        """
        :lambda_val = Regularization parameter
        :nu_val = number of boosting rounds
        :sigma_sq_val = Typically 1. Value only matters relative to value of lambda.
        """
        assert lambda_val > 0.0
        assert nu_val > 0
        assert sigma_sq_val > 0
        self.lambda_val = lambda_val
        self.nu_val = nu_val
        self.sigma_sq_val = sigma_sq_val
        self.theta_hat_boost = None
    
    def fit(self, X, y, K=None):
        """
        """
        if K is None:
            K = np.identity(X.shape[1])
        
        theta_hat_list = []
        y_res_list = []

        theta_hat_list.append(self.lambda_val*K @ X.T @ np.linalg.inv( self.lambda_val * X @ K @ X.T + np.identity(X.shape[0])*self.sigma_sq_val) @ y)
        y_hat = X @ theta_hat_list[0]
        y_res_list.append(y - y_hat)

        for boost_round in range(1, self.nu_val):
            theta_hat_list.append(self.lambda_val*K @ X.T @ np.linalg.inv( self.lambda_val * X @ K @ X.T + np.identity(X.shape[0])*self.sigma_sq_val) @ y_res_list[boost_round-1])
            y_res_list.append(y_res_list[boost_round-1] - X @ theta_hat_list[boost_round])

        self.theta_hat_boost = np.array(theta_hat_list).sum(axis=0)
        
    def predict(self, X):
      return X @ self.theta_hat_boost

class L2KernelBoost:
    """
    """
    def __init__(self, lambda_val=0.1, nu_val=0.2, sigma_sq_val=1):
        """
        :lambda_val = Regularization parameter
        :nu_val = number of boosting rounds
        :sigma_sq_val = Typically 1. Value only matters relative to value of lambda.

        """
        assert lambda_val > 0.0
        assert nu_val > 0
        assert sigma_sq_val > 0
        self.lambda_val = lambda_val
        self.nu_val = nu_val
        self.sigma_sq_val = sigma_sq_val
        self.theta_hat_boost = None
    
    def fit(self, X, y, K=None):
        """
        """
        if K is None:
          K = np.identity(X.shape[1])
        
        V, D, _ = np.linalg.svd(X @ K @ X.T) 

        self.theta_hat_boost = np.linalg.inv(X.T @ X) @ X.T @ V @ (
                np.identity(X.shape[0])
                - np.diag(np.power((self.lambda_val*D+self.sigma_sq_val*np.ones(D.shape[0]))/self.sigma_sq_val, -1*self.nu_val))
                ) @ V.T @ y
        
    def predict(self, X):
      return X @ self.theta_hat_boost

class ExplicitKernelBoost:
    """
    """
    def __init__(self, loss_type, lambda_val=0.1, nu_val=0.2, sigma_sq_val=1, loss_term_kwarg={'M':1}, solver=cp.SCS):
        """
        :lambda_val = Regularization parameter
        :nu_val = number of boosting rounds
        :sigma_sq_val = Typically 1. Value only matters relative to value of lambda.

        """
        assert lambda_val > 0.0
        assert nu_val > 0
        assert sigma_sq_val > 0
        self.lambda_val = lambda_val
        self.nu_val = nu_val
        self.sigma_sq_val = sigma_sq_val
        self.theta_hat_boost = None
        self.loss_func_type = loss_type
        self.loss_term_kwarg = loss_term_kwarg
        self.solver=cp.ECOS
    
    def fit(self, X, y, K=None):
        """
        """
        if K is None:
            K = np.identity(X.shape[1])

        theta_cvx = cp.Variable(shape=(X.shape[1],1))
        V, D, _ = np.linalg.svd(X @ K @ X.T) 
        Reg_Mat = X.T @ V @ np.diag(np.power(np.power(D*self.lambda_val/self.sigma_sq_val+1, self.nu_val )-np.ones(D.shape[0])+0.0000001,-1)) @ V.T @ X

        if self.loss_func_type == 'square':
            loss_term = cp.sum_squares(y - X @ theta_cvx) 
        elif self.loss_func_type == 'absolute':
            loss_term = cp.sum(cp.abs(y - X @ theta_cvx))
        elif self.loss_func_type == 'hinge':
            loss_term = cp.sum(cp.pos(1- cp.multiply(y, X @ theta_cvx)))
        elif self.loss_func_type == 'vapnik':
            loss_term = cp.sum(cp.max(cp.abs(y - X @ theta_cvx)-self.loss_term_kwarg['M'],0))
        elif self.loss_func_type == 'huber':
            loss_term = cp.sum(cp.huber(y - X @ theta_cvx, M=self.loss_term_kwarg['M']))

        optimzer_term = loss_term + cp.quad_form(theta_cvx, Reg_Mat)
        objective = cp.Minimize(optimzer_term)
        prob = cp.Problem(objective)
        prob.solve(solver=self.solver, kktsolver=cp.ROBUST_KKTSOLVER)

        self.theta_hat_boost = theta_cvx.value
        
    def predict(self, X):
      return X @ self.theta_hat_boost

class KernelBoost:

    def __init__(self, kernel_func, loss_type, lambda_val=0.1, nu_val=10, sigma_sq_val=1, loss_term_kwarg={'M':5}, solver=cp.SCS):
        """
        :lambda_val = Regularization parameter
        :nu_val = number of boosting rounds
        :sigma_sq_val = Typically 1. Value only matters relative to value of lambda.

        """
        assert lambda_val > 0.0
        assert nu_val > 0
        assert sigma_sq_val > 0
        self.lambda_val = lambda_val
        self.nu_val = nu_val
        self.sigma_sq_val = sigma_sq_val
        self.kernel_func = kernel_func
        self.loss_func_type = loss_type
        self.loss_term_kwarg = loss_term_kwarg
        self.theta_hat_boost = None
        self.solver = solver
        self.X = None

    def fit(self, X, y):
        """
        """
        self.X = X 

        K = self.kernel_func(self.X, self.X)
        V, D, _ = np.linalg.svd(K)
        Reg_Mat = self.sigma_sq_val*V @ ( 
            np.diag(np.power(D*self.lambda_val/self.sigma_sq_val+1, self.nu_val ) - np.ones(D.shape[0])+0.00000001) #0.00..1 added for numerical stability
                      ) @ V.T

        b_cvx = cp.Variable(shape=(X.shape[0],1))
        regularization_term = cp.quad_form(b_cvx, Reg_Mat)

        if self.loss_func_type == 'square':
            loss_term = cp.sum_squares(y - Reg_Mat @ b_cvx) 
        elif self.loss_func_type == 'absolute':
            loss_term = cp.sum(cp.abs(y - Reg_Mat @ b_cvx))
        elif self.loss_func_type == 'hinge':
            loss_term = cp.sum(cp.pos(1- cp.multiply(y , Reg_Mat @ b_cvx)))
        elif self.loss_func_type == 'vapnik':
            loss_term = cp.sum(cp.max(cp.abs(y - Reg_Mat @ b_cvx)-self.loss_term_kwarg['M']))
        elif self.loss_func_type == 'huber':
            loss_term = cp.sum(cp.huber(y - Reg_Mat @ b_cvx, M=self.loss_term_kwarg['M']))
        
        optimzer_term = loss_term + regularization_term
        objective = cp.Minimize(optimzer_term)
        prob = cp.Problem(objective)
        prob.solve(solver=self.solver, kktsolver=cp.ROBUST_KKTSOLVER)
        self.theta_hat_boost = np.linalg.pinv(K) @ Reg_Mat @ b_cvx.value # b_cvx.value

    def predict(self, X):
        """
        """
        K = self.kernel_func(X, self.X)
        y_pred = K @ self.theta_hat_boost
        return y_pred
    
class IterativeKernelBoost:

    def __init__(self, kernel_func, loss_type, lambda_val=0.1, nu_val=10, sigma_sq_val=1, loss_term_kwarg={'M':5}, solver=cp.SCS):
        """
        :lambda_val = Regularization parameter
        :nu_val = number of boosting rounds
        :sigma_sq_val = Typically 1. Value only matters relative to value of lambda.

        """
        assert lambda_val > 0.0
        assert nu_val > 0
        assert sigma_sq_val > 0
        self.lambda_val = lambda_val
        self.nu_val = nu_val
        self.sigma_sq_val = sigma_sq_val
        self.kernel_func = kernel_func
        self.loss_func_type = loss_type
        self.loss_term_kwarg = loss_term_kwarg
        self.theta_hat_boost = None
        self.solver = solver
        self.X = None

    def fit(self, X, y):
        """
        """
        self.X = X 
        theta_hat_boost_list = []
        K = self.kernel_func(self.X, self.X)
        K_reg = K * (self.sigma_sq_val/self.lambda_val)

        b_cvx = cp.Variable(shape=(X.shape[0],1))
        regularization_term = cp.quad_form(b_cvx, K_reg)

        if self.loss_func_type == 'square':
            loss_term = cp.sum_squares(y - K @ b_cvx) 
        elif self.loss_func_type == 'absolute':
            loss_term = cp.sum(cp.abs(y - K @ b_cvx))
        elif self.loss_func_type == 'hinge':
            loss_term = cp.sum(cp.pos(1 - cp.multiply(y , K @ b_cvx)))
        elif self.loss_func_type == 'vapnik':
            loss_term = cp.sum(cp.max(cp.abs(y - K @ b_cvx)-self.loss_term_kwarg['M']))
        elif self.loss_func_type == 'huber':
            loss_term = cp.sum(cp.huber(y - K @ b_cvx, M=self.loss_term_kwarg['M']))
        
        optimzer_term = loss_term + regularization_term
        objective = cp.Minimize(optimzer_term)
        prob = cp.Problem(objective)
        prob.solve(solver=self.solver, kktsolver=cp.ROBUST_KKTSOLVER)
        theta_hat_boost_list.append(b_cvx.value)
        y_res = y - K @ b_cvx.value

        del b_cvx
        del regularization_term
        del loss_term
        del optimzer_term

        for _ in range(1, self.nu_val):
            b_cvx = cp.Variable(shape=(X.shape[0],1))
            regularization_term = cp.quad_form(b_cvx, K_reg)

            if self.loss_func_type == 'square':
                loss_term = cp.sum_squares(y_res - K @ b_cvx)
            elif self.loss_func_type == 'absolute':
                loss_term = cp.sum(cp.abs(y_res - K @ b_cvx))
            elif self.loss_func_type == 'hinge':
                loss_term = cp.sum(cp.pos(1- cp.multiply(y_res , K @ b_cvx)))
            elif self.loss_func_type == 'vapnik':
                loss_term = cp.sum(cp.max(cp.abs(y_res - K @ b_cvx)-self.loss_term_kwarg['M']))
            elif self.loss_func_type == 'huber':
                loss_term = cp.sum(cp.huber(y_res - K @ b_cvx, M=self.loss_term_kwarg['M']))

            optimzer_term = loss_term + regularization_term
            objective = cp.Minimize(optimzer_term)
            prob = cp.Problem(objective)
            prob.solve(solver=self.solver, kktsolver=cp.ROBUST_KKTSOLVER)
            theta_hat_boost_list.append(b_cvx.value)
            y_res = y_res - K @ b_cvx.value

            del b_cvx
            del regularization_term
            del loss_term
            del optimzer_term

        self.theta_hat_boost = np.array(theta_hat_boost_list).sum(axis=0)

    def predict(self, X):
        """
        """
        K = self.kernel_func(X, self.X)
        y_pred = K @ self.theta_hat_boost
        return y_pred

def main():
    return None

if __name__=='__main__':
    main()