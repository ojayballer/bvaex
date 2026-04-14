import jax.numpy as jnp

class ELBO:
    def  __init__(self):
        self.RL_object=ReconstructionLoss()
        self.KL_object=KullBackLeiblergDivergenceLoss()
        
    def forward(self,y_true,y_pred,mu,log_var,kl_weight): # original image,reconstructed image
        self.y_true=y_true
        self.y_pred=y_pred
        self.mu=mu
        self.log_var=log_var
        self.kl_weight=kl_weight
        
        self.rl_loss=self.RL_object.forward(self.y_pred,self.y_true)
        self.kl_loss=self.KL_object.forward(self.mu,self.log_var)
        self.kl_weight=kl_weight
        
        return self.rl_loss+(self.kl_weight*self.kl_loss)   #ELBO loss=Kl loss+Reconstruction loss, Beta VAE
        
    #update:after runnning 1 epoch,i noticed the Kl loss was only reducing and he RL loss was flunctuating,in order to make training stable,I have devided to add KL-annealing,
    #This alllows the model to focus  on reconstructing images(RL loss) first and then when RL gets satble,KL loss also kicks in and starts improving over the latter epochs
    def backward(self):
        self.mu_gradient,self.log_var_gradient=self.KL_object.backward()
        return self.RL_object.backward(),self.kl_weight*self.mu_gradient,self.kl_weight*self.log_var_gradient

class ReconstructionLoss:   #for the reconstruction LOss ,I want to use Mean squared error 
    def __init__(self):
        pass
        
    def forward(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true
        return jnp.mean(jnp.sum((self.y_true-self.y_pred)**2, axis=(1, 2, 3)))  
        
    def backward(self):
        return 2*(self.y_pred - self.y_true) / self.y_true.shape[0]
        
class KullBackLeiblergDivergenceLoss :
    def __init__(self):
        pass
        
    def forward(self,mu,log_var):
        self.mu=mu
        self.log_var=jnp.clip(log_var, -10, 10) 
        return -0.5 * jnp.mean(jnp.sum(1 + self.log_var - self.mu**2 - jnp.exp(self.log_var), axis=1))  
        
    def backward(self):
        batch_size = self.mu.shape[0]
        return self.mu / batch_size, 0.5*(jnp.exp(self.log_var)-1) / batch_size