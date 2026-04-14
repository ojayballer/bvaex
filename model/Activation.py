import jax.numpy as jnp
class Activation:
    def __init__(self,activation_object):
       self.activation_object=activation_object

    def forward(self,input):
        self.input=input
        return self.activation_object.forward(self.input)
    
    def backward(self,output_gradient):
         self.output_gradient=output_gradient
         return self.activation_object.backward(self.output_gradient)

class RELU:
    def __init__(self):
        pass

    def  forward(self,input):
        self.input=input
        return jnp.maximum(0,self.input)
    
    def backward(self,output_gradient):  #relu prime
        self.output_gradient=output_gradient
        return self.output_gradient*(self.input>0 )

class Sigmoid :
    def __init__(self):
       pass

    def forward(self,input):
        self.input=input
        self.sigmoid= 1/(1+jnp.exp(-self.input))
        return self.sigmoid
    
    def backward(self,output_gradient): 
        self.output_gradient=output_gradient
        return self.output_gradient*(self.sigmoid *(1-self.sigmoid))