import jax
import jax.numpy as jnp

class Adam:
    def __init__(self):
        self.alpha = 0.001
        self.B1 = 0.9
        self.B2 = 0.999
        self.epsilon = 1e-8
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, name, weight, gradient):
        if name not in self.m:
            self.m[name] = jnp.zeros_like(weight)
            self.v[name] = jnp.zeros_like(weight)

        # Weight decay
        weight = weight * (1.0 - self.alpha * 0.01)

        self.m[name] = self.B1 * self.m[name] + (1 - self.B1) * gradient
        self.v[name] = self.B2 * self.v[name] + (1 - self.B2) * gradient**2

        m_hat = self.m[name] / (1 - self.B1**self.t)
        v_hat = self.v[name] / (1 - self.B2**self.t)

        return weight - self.alpha * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

    def step(self):
        self.t += 1