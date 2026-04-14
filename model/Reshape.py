import jax.numpy as jnp
class Reshape :
    def __init__(self,input_):
        self.original_input_shape=None

    def forward(self, input):
        self.input=input
        self.original_input_shape=input.shape
        return jnp.reshape(self.input,(self.input.shape[0],-1))
    

    def backward(self,output_gradient):
        self.output_gradient=output_gradient
        return jnp.reshape(self.output_gradient ,self.original_input_shape)