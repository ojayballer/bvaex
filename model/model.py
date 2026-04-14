from .ELBO import ELBO
from .Encoder import Encoder,Convolutional
from .Decoder import Decoder,TransposedConv2D
from .Activation import Activation
from .Reshape import Reshape as flatten
from .Reparameterize import Reparameterize 
import jax.numpy as jnp
from .Adam import Adam

class CONVAE:
    def __init__(self,input,input_shape,kernel_size,encoder_filters,decoder_filters,latent_dim):
        #wow!!! I was actually  thinking 10 steps ahead when I was writing the building blocks,everything is sooo easy to string togther ,no hardcoding !!!,LFG!!!
        self.adam=Adam()
        self.encoder=Encoder(encoder_filters,input,kernel_size,latent_dim,self.adam) 
        self.decoder=Decoder(input_shape,decoder_filters,latent_dim,self.adam)
        self.reparamterize=Reparameterize()
        self.elbo=ELBO()
        
            
    def forward(self,input,key,kl_weight):
        self.input=input
        self.mu,self.log_var=self.encoder.forward(self.input) #x->encoder->relu->mu and log_var
        self.z=self.reparamterize.forward(key,self.mu,self.log_var) #mu,log_var->z
        self.reconstructed_image=self.decoder.forward(self.z)  #z->dense->reshape to 3D->(transposed_conv,RELU,transposed_conv)->sigmoid->x'(note: the last transposed conv cannot be passed to relu,it 
                                                                #must only be passed into sigmoid)
        self.loss=self.elbo.forward(self.input,self.reconstructed_image,self.mu,self.log_var,kl_weight)
        return self.loss,self.elbo.rl_loss,self.elbo.kl_loss
        # return loss-> forward(RL+KL) ,RL_loss.forward(),KL_loss.forward()
        
    def backward(self,learning_rate):
        self.learning_rate=learning_rate
        self.RL_gradient,self.mu_gradient,self.log_var_gradient=self.elbo.backward()
        self.z_input_gradient=self.decoder.backward(self.RL_gradient,self.learning_rate) #c update te weights and biases in the transposedconv2d class

        self.total_mu_gradient,self.total_log_var_gradient=self.reparamterize.backward(self.z_input_gradient)
        self.total_mu_gradient=self.mu_gradient+self.total_mu_gradient

        self.total_log_var_gradient=self.log_var_gradient+self.total_log_var_gradient # add the twp gradients to reconstructthe input gradient ,nd we will update the weigts

        self.input_gradient=self.encoder.backward(self.total_mu_gradient,self.total_log_var_gradient,learning_rate)










         