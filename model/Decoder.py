from .Dense import DenseLayer as Dense 
import jax.numpy as jnp 
from .Reshape import Reshape as flatten
import jax
from scipy import signal
from .Activation import Activation,RELU,Sigmoid

class TransposedConv2D:
    def __init__(self,input_shape,depth,adam,kernel_size=4 ,padding=1,stride=2,seed=1):
#->z(z is a dense layer already but it has small features,we can make it richer with the next step)>dense layer.forward(we want more features,calling .forwrd on z 
# expands z by multiplying it with initiaoed wegths,this gives us larger features to pass into the transposed conv layer)->we have tp reshape the dense layer
#  to the 3D format that the transposed conv is expecting->transposed conv again and then we use an activation function to output the reconstructed x of the latent 
# space representation.

         input_depth,input_height,input_width=input_shape
         self.input_depth=input_depth #filter no for prev layer
         self.input_height=input_height
         self.input_width=input_width
         self.input_shape=input_shape
         self.kernel_size=kernel_size
         self.depth=depth #filter number for current layer 
         self.padding=padding
         self.stride=stride
         
         # Exact formula for Transposed Conv Output Size (updated for kernel_size=4, padding=1)
         self.output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size
         self.output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size

         self.output_shape=(self.depth,self.output_height,self.output_width)

         self.kernel_shape=(self.input_depth,depth,self.kernel_size,self.kernel_size) # in
         
         #Xavier/Glorot Iitialization
         key = jax.random.PRNGKey(seed)
         wkey, bkey = jax.random.split(key)
         fan_in = input_depth * self.kernel_size * self.kernel_size
         fan_out =self. depth * self.kernel_size * self.kernel_size
         std = jnp.sqrt(2.0 / (fan_in + fan_out))
            
         self.weights=jax.random.normal(wkey,self.kernel_shape) *std
         self.biases=jnp.zeros(self.output_shape)

         self.dn = ('NCHW', 'OIHW', 'NCHW')
         self.adam=adam
         


    def forward(self,input):
         self.input=input #->(batch_size,depth,height,wdth)

         # Calculate necessary padding for the JAX dilated conv
         pad = self.kernel_size - 1 - self.padding
         
         # Swap Input/Output channels and flip spatial dimensions for forward transposed pass
         # This is the correct formulation that makes the backward pass work properly with kernel_size=4
         w_flipped = jnp.flip(self.weights, axis=(2, 3))
         w_forward = jnp.transpose(w_flipped, (1, 0, 2, 3))
         
         #I switched to jax.lax.conv for fater training snd computatin of matrices 
         output = jax.lax.conv_general_dilated(
             self.input, w_forward,
             window_strides=(1, 1),
             padding=[(pad, pad), (pad, pad)],
             lhs_dilation=(self.stride, self.stride),
             dimension_numbers=self.dn
         )
         output = output + self.biases[None, :, :, :]
         return output
              



    def  backward(self,output_gradient,learning_rate):
          
          batch_size = output_gradient.shape[0]
          
          # 1. Input Gradient (dX) -> Backward of Transposed Conv is a Regular Conv
          input_gradient = jax.lax.conv_general_dilated(
              output_gradient, self.weights,
              window_strides=(self.stride, self.stride),
              padding=[(self.padding, self.padding), (self.padding, self.padding)],
              dimension_numbers=self.dn
          )
          
          # 2. Kernel Gradient (dW) -> Cross-correlation between dilated input and dY
          input_t = jnp.transpose(self.input, (1, 0, 2, 3)) 
          grad_t = jnp.transpose(output_gradient, (1, 0, 2, 3)) 
          pad = self.kernel_size - 1 - self.padding
          kernels_gradient = jax.lax.conv_general_dilated(
              input_t, grad_t,
              window_strides=(1, 1),
              padding=[(pad, pad), (pad, pad)],
              lhs_dilation=(self.stride, self.stride),
              dimension_numbers=('NCHW', 'OIHW', 'NCHW')
          )
          
          # Un-flip the gradient to match original weight orientation
          kernels_gradient = jnp.flip(kernels_gradient, axis=(2, 3))
          kernels_gradient = kernels_gradient / batch_size

          # Before the adam update
          kernels_gradient = jnp.clip(kernels_gradient, -1.0, 1.0)

          # 3. Bias Gradient
          bias_gradient = jnp.sum(output_gradient, axis=0) / batch_size
          bias_gradient = jnp.clip(bias_gradient, -1.0, 1.0)

          #Using Adam
          #I had to switch to another optimizer,training was very slow and my KL loss kept exploding and vaniishing,
          #I couldn't find the best hyperparameters for SGD ,so I switched to adam to accomodate that and also let me achieve convergence fatser
            
          self.weights=self.adam.update(f"{id(self)}_weights",self.weights,kernels_gradient)
          self.biases = self.adam.update(f"{id(self)}_biases",self.biases,bias_gradient)


          return input_gradient



class Decoder :
     def __init__(self,input_shape,filters,latent_dim,adam):  #in my next project ,I would be very careful about tensors and be strict about the shapes I wuld pass
          # into my layers,JAX doesn't really do too well if combined with oop,they are two different types of programming
          
          self.input_shape=input_shape  #last otuput of my encoder's last conv layer ,the decoder is trying to reconstruct the input image,so it has to go in reverse lol
          self.filters=filters
          self.latent_dim=latent_dim  #for clarity ->latent dim is a tuple in the format (x,y)

          
          self.transposed_conv_layers=[]
          
          self.flatten_current_input_shape=input_shape[0]*input_shape[1]*input_shape[2]
          self.dense_layer=Dense(adam,(self.latent_dim,),self.flatten_current_input_shape)


       
          next_input=self.input_shape  #0,1,2
          for i in range(len(self.filters)):
               layer=TransposedConv2D(next_input,self.filters[i],adam,seed=i+10)

               if (i == len(self.filters)-1): # if we are at the last conv layer ,we wnt  to apply sigmoid instead of RELU
                   layer.activation=Activation(Sigmoid())
               else :
                   layer.activation=Activation(RELU())  #else,we apply RELU 

               self.transposed_conv_layers.append(layer)
               next_input=layer.output_shape
 
          
                                     

     def forward(self,z):
          self.z=z
          self.z=self.dense_layer.forward(self.z)
          self.z=jnp.reshape(self.z,((self.z.shape[0],)+self.input_shape)) #->(batch,depth,height,width)
          next_input=self.z
          for i in range(len(self.filters)):
               self.z_output=self.transposed_conv_layers[i].forward(next_input)
               self.z_output=self.transposed_conv_layers[i].activation.forward(self.z_output)
               next_input=self.z_output
        
         
          return self.z_output   #I would haave to pass each z_ouput into an activation function like  RElu ,to introduce non-linearity,
                              #The last z_output would have to be passed into another dense layer and then into a sigmoid fuction to get the reconstructed ->x
     
     def backward(self,output_gradient,learning_rate):
          self.output_gradient=output_gradient
          self.learning_rate=learning_rate

          for i in reversed(range(len(self.transposed_conv_layers))) :  #x->RELU->transconv->RELU->transconv
               self.output_gradient=self.transposed_conv_layers[i].activation.backward(self.output_gradient)

               self.input_gradient=self.transposed_conv_layers[i].backward(self.output_gradient,self.learning_rate,)
               self.output_gradient=self.input_gradient


          self.input_gradient = jnp.reshape(self.input_gradient, (self.input_gradient.shape[0], -1))
          self.z_input_gradient = self.dense_layer.backward(self.input_gradient, learning_rate)
          

          return self.z_input_gradient #when backpropagating I would also have to pass each input gradient into a relu function before
          #passing it into the transposed conv layers in the decoder .