import jax.numpy as jnp
import jax
class Reparameterize:
    def __init__(self):
        
        self.epilson=None
        pass
    
    def forward(self,key,mu,log_var):
        self.key=key
        self.mu=mu
        self.log_var=log_var
        self.sigma=jnp.exp(0.5*self.log_var)
        self.epsilon=jax.random.normal(key,self.sigma.shape)

        return self.mu + self.sigma*self.epsilon  #z
        
    def backward(self, output_gradient):
        total_mu_gradient = output_gradient  # dL/dz * dz/dmu = output_gradient * 1
        total_log_var_gradient = output_gradient * 0.5 * self.sigma * self.epsilon  # dL/dz * dz/dlog_var
        return total_mu_gradient, total_log_var_gradient

