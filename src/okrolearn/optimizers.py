from okrolearn.okrolearn import np


class AdamOptimizer:
    # Generate doctstring codeium
    """
    Parameters
    ----------
    lr : float
        Learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates
    beta2 : float
        Exponential decay rate for the second moment estimates
    epsilon : float
        Constant for numerical stability
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, grad, key):
        self.t += 1

        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        layer -= update

    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0


class NadamOptimizer:
    # Simmilar to Adam Optimizer
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, grad, key):
        self.t += 1

        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        beta1_t = self.beta1 * (1 - self.beta1 ** (self.t - 1)) / (1 - self.beta1 ** self.t)
        beta2_t = self.beta2 * (1 - self.beta2 ** (self.t - 1)) / (1 -
                                                                   self.beta2 ** self.t)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        m_nesterov = beta1_t * m_hat + (1 - beta1_t) * grad

        update = self.lr * m_nesterov / (np.sqrt(v_hat) + self.epsilon)

        layer -= update

    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0


class AdadeltaOptimizer:
    """
    Parameters
    ----------
    rho : float
        Decay rate for the first moment estimates
    epsilon : float
        Constant for numerical stability
    """

    def __init__(self, rho=0.95, epsilon=1e-8):
        self.rho = rho
        self.epsilon = epsilon
        self.E_g2 = {}
        self.E_dx2 = {}

    def update(self, layer, grad, key):
        if key not in self.E_g2:
            self.E_g2[key] = np.zeros_like(grad)
            self.E_dx2[key] = np.zeros_like(grad)

        self.E_g2[key] = self.rho * self.E_g2[key] + (1 - self.rho) * (grad ** 2)

        RMS_g = np.sqrt(self.E_g2[key] + self.epsilon)
        RMS_dx = np.sqrt(self.E_dx2[key] + self.epsilon)
        update = (RMS_dx / RMS_g) * grad

        self.E_dx2[key] = self.rho * self.E_dx2[key] + (1 - self.rho) * (update ** 2)

        layer -= update

    def reset(self):
        self.E_g2 = {}
        self.E_dx2 = {}


class SGDOptimizer:
    """
    Parameters:
    -----------
    self lr: learning rate
    self momentum: momentum
    self velocity: velocity
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, layer, grad, key):
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(grad)
        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
        layer += self.velocity[key]

        layer.grad = np.zeros_like(grad)
        layer.backward_fn = None


class RMSPropOptimizer:
    """
    Parameters
    ----------
    lr : float
        Learning rate
    beta : float
        Exponential decay rate for the second moment estimates
    epsilon : float
        Constant for numerical stability
    """

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.v = {}

    def update(self, layer, grad, key):
        if key not in self.v:
            self.v[key] = np.zeros_like(grad)

        self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (grad ** 2)

        update = self.lr * grad / (np.sqrt(self.v[key]) + self.epsilon)
        layer -= update

    def reset(self):
        self.v = {}
