import jax.numpy as jnp
import jax
class DenseLayer :
        def  __init__(self,adam,input_shape ,output_shape,seed=42 ):
            self.input_shape=input_shape
            self.output_shape=output_shape

            

            input_size=input_shape[0]  #tuple indexing
            output_size=output_shape   #we just need to pass the actual latent_dim value into the output size
            key = jax.random.PRNGKey(seed)
            #to get new seeds so the network can learn better
            nk,ok=jax.random.split(key)

            # Xavier/Glorot initialization
            std = jnp.sqrt(2.0 / (input_size + output_size))# squish factor  -> 2/(fan in+fan out ) where fan in is the number of input units and 
            #fan out is the number of output units
            self.weight = jax.random.normal(nk, (input_size, output_size)) * std # normalize the weights and multiply by the squish factor 

            self.bias=jnp.zeros(self.output_shape)

            #adam
            self.adam=adam
            

        def forward(self,input):
            self.input=input 

            return jnp.dot(self.input,self.weight)+ self.bias

        def backward(self,output_gradient,learning_rate):
            
            weights_gradient=jnp.dot(self.input.T,output_gradient)  #the gradient of the weights is the dot product of the input and the output gradient
            input_gradients=jnp.dot(output_gradient,self.weight.T)
            bias_gradients=jnp.sum(output_gradient,axis=0)

            weights_gradient = jnp.clip(weights_gradient, -1.0, 1.0)
            bias_gradients = jnp.clip(bias_gradients, -1.0, 1.0)
     

            #update the parametrs,I am using the ADAm optimizer for some reasons,I mentioneed it in my encoder ad decoder files
            self.weight=self.adam.update(f"{id(self)}_dense_weights",self.weight,weights_gradient)
            self.bias=self.adam.update(f"{id(self)}_dense_biases",self.bias,bias_gradients)

            return input_gradients