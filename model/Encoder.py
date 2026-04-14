import jax.numpy as jnp 
from scipy import signal
from .Dense import DenseLayer as Dense 
from .Reshape import Reshape as flatten
import jax
from scipy import signal
from .Activation import Activation,RELU


class Convolutional:
        def __init__(self, input_shape,adam, kernel_size, depth, padding=1, stride=2,seed=0):
            input_depth, input_height, input_width = input_shape
            self.input_shape = input_shape
            self.kernel_size = kernel_size
            self.input_depth = input_depth #depth channel
            self.depth = depth  # filter_no
            
            self.padding = padding
            self.stride = stride

            # H_out = floor((H + 2P - F) / S) + 1
            # W_out = floor((W + 2P - F) / S) + 1
            self.output_height = int(jnp.floor((input_height + 2 * self.padding - self.kernel_size) / self.stride) + 1)
            self.output_width = int(jnp.floor((input_width + 2 * self.padding - self.kernel_size) / self.stride) + 1)

            self.output_shape = (depth, self.output_height, self.output_width)

            # weights(kernels), shape=(filter_no, input_depth, kernel_size, kernel_size)
            self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)

            key = jax.random.PRNGKey(seed)
            wkey, bkey = jax.random.split(key)

            #Xavier/Glorot Initialization
            fan_in = input_depth * self.kernel_size * self.kernel_size
            fan_out =self. depth * self.kernel_size * self.kernel_size
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            

            self.weights = jax.random.normal(wkey, self.kernel_shape) *std
            self.biases = jnp.zeros(self.output_shape)

            self.dn = ('NCHW', 'OIHW', 'NCHW') # I might have to switch to jax.lax.conv ,scipy.signal is making training painstakingly slow ad brutal
            self.adam=adam

        def forward(self, input):
            self.input = input
            
            #this leverages scipy.signal,but I had to switch beacuse it was making training very slow
            '''
            batch_size = input.shape[0]
            # Create output with batch dimension: (batch, depth, height, width)
            self.output = jnp.zeros((batch_size,) + self.output_shape)
            
            # Loop over batch
            for b in range(batch_size):
                for i in range(self.depth):  # filter i
                    for j in range(self.input_depth):  # depth channel
                        
                        # Calculate cross-correlation on the b-th sample
                        # input[b, j] gives us a 2D array (H, W)
                        correlation = signal.correlate2d(self.input[b, j], self.weights[i, j], mode='valid') #filter *input _mtatrix forbatch i-th
                        correlation = correlation[::self.stride, ::self.stride] # apply stride 

                        # Add to the i-th output map for batch b
                        self.output = self.output.at[b, i].add(correlation)
                    
                    # Add bias to the i-th output map for batch b
                    self.output = self.output.at[b, i].add(self.biases[i])
            
            return self.output'''
            
            #this uses jax.lax.conv,this way I can leverage XLA and make training faster on GPU
            output = jax.lax.conv_general_dilated(self.input,self.weights, window_strides=(self.stride, self.stride),padding=[(self.padding, self.padding), 
            (self.padding, self.padding)],   dimension_numbers=self.dn)
            output = output + self.biases[None, :, :, :]
            return output

            
             

        
        def backward(self, output_gradient, learning_rate):
            batch_size = output_gradient.shape[0]

            #same issue here..
            '''  
            kernels_gradient = jnp.zeros(self.kernel_shape)
            input_gradient = jnp.zeros(self.input.shape)

            # Loop over batch
            for b in range(batch_size):
                for i in range(self.depth):  # Loop through each Filter (i)
                    for j in range(self.input_depth):  # each input channel/depth

                        # upsample output_gradient back to full size before computing gradients
                        og = output_gradient[b, i]
                        upsampled = jnp.zeros((og.shape[0] * self.stride - (self.stride-1),og.shape[1] * self.stride - (self.stride-1)))
                        upsampled = upsampled.at[::self.stride, ::self.stride].set(og)
                       

                    
                        #  Calculate Kernel Gradient (dK)
                        # dE/dKij = Xj ⋆ dE/dYi
                        correlation = signal.correlate2d(self.input[b, j], upsampled, "valid")
                        kernels_gradient = kernels_gradient.at[i, j].add(correlation)

                        # Calculate Input Gradient (dX)
                        # dE/dX = dE/dY ∗ full K
                        convolution = signal.convolve2d(upsampled, self.weights[i, j], "full")
                        input_gradient = input_gradient.at[b, j].add(convolution)

            # Average gradients over batch
            kernels_gradient = kernels_gradient / batch_size
            '''


            # Kernel gradient
            input_t = jnp.transpose(self.input, (1, 0, 2, 3))  # (batch, C_in, H, W) -> (C_in, batch, H, W)
            grad_t = jnp.transpose(output_gradient, (1, 0, 2, 3)) # (batch, C_out, oH, oW) -> (C_out, batch, oH, oW)
            kernels_gradient = jax.lax.conv_general_dilated(input_t, grad_t,window_strides=(1, 1),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],rhs_dilation=(self.stride, self.stride),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')   )
    
            kernels_gradient = jnp.transpose(kernels_gradient, (1, 0, 2, 3)) / batch_size
            # Before the adam update
            kernels_gradient = jnp.clip(kernels_gradient, -1.0, 1.0)

            # Input gradient
            flipped = jnp.flip(self.weights, axis=(2, 3))
            flipped = jnp.transpose(flipped, (1, 0, 2, 3))
            pad = self.kernel_size - 1 - self.padding
            input_gradient = jax.lax.conv_general_dilated(output_gradient, flipped,window_strides=(1, 1),padding=[(pad, pad), (pad, pad)],
                                                          lhs_dilation=(self.stride, self.stride), dimension_numbers=self.dn)

            
            # Update Weights,using SGD
            #self.weights = self.weights - learning_rate * kernels_gradient

            
            #Using Adam
            #I had to switch to another optimizer,training was very slow and my KL loss kept exploding and vaniishing,
            #I couldn't find the best hyperparameters for SGD ,so I switched to adam to accomodate that and also let me achieve convergence fatser
            
            self.weights=self.adam.update(f"{id(self)}_weights",self.weights,kernels_gradient)
            
            
            # Update Biases 
            bias_gradient = jnp.sum(output_gradient, axis=0) / batch_size
            bias_gradient = jnp.clip(bias_gradient, -1.0, 1.0)
            self.biases = self.adam.update(f"{id(self)}_biases",self.biases,bias_gradient)

            return input_gradient
        
        
class Encoder():
        def __init__(self,filters,x,kernel_size,latent_dim,adam):
            self.filters=filters
            self.x=x
            self.kernel_size=kernel_size
            self.latent_dim=latent_dim # to set the outputshape of the dense layers that genrate mu and var
             
            

            self.conv_layers=[]
            #dynamic and not hardcoded, the no of filters indicate how many conv layer objects I want to create ,so if i.e filters=[32,64,128],
            # it means I wnat to use 3 conv layers ,this removes the staticity of my encoder class(I rewrote it twice) leading to more code control 
            # and better readablity 

            current_input_shape=self.x #the first cnonv layer input shape is the orginal input shape of x
            for i in range(len(self.filters)):
                #filter_no/depth(according to the original Convolutional class) is the number of filters in  the conv layer 
                layer=Convolutional(current_input_shape,adam,kernel_size=self.kernel_size,depth=self.filters[i],seed=i)
                layer.activation=Activation(RELU()) #   #I need to introduce non-linearity to my model,I discovered I forgot  to add this when I 
                          #was writing the training loop in training.py lol(one layer object for each convolutional object )

                self.conv_layers.append(layer)
                current_input_shape=layer.output_shape #the output shape of the conv layer is the input shape of the next conv layer

            
            self.Flatten=flatten(current_input_shape) #create the flatten objec to be used in the forward and backward method

            
            flatten_current_input_shape=current_input_shape[0]*current_input_shape[1]*current_input_shape[2]# i have to flatten before passing it to the dense layer
            #or else JAX would crash it 

            #I also want to wrap it in a tuple to be safe when I pass the actual tuples for the DEnse Layer when training  
            
            self.mu=Dense(adam,(flatten_current_input_shape,),latent_dim)
            self.log_var=Dense(adam,(flatten_current_input_shape,),latent_dim)

                  #sighs
            
        
        #The rael stuff happens here ,sorry if my code is complex to understand !
        def forward(self,x):
            for i in  range(len(self.conv_layers)): #  for layer in layer #the putput of one  conv layer is the input of the next conv layer 
                x=self.conv_layers[i].forward(x)   #yh let's gooo!!!!!!
                x=self.conv_layers[i].activation.forward(x)  #[layer.activation.forward()] ,apply activation after each convolutional layer ,this introduces non-linearity,Relu preserves the 
                                                 #3D shape so I don;t hve to worry about that

            ''' x is the output of each  conv layer and it is the input of the next conv layer and
            #the last conv kayer output is the final output of the encoder class used to generate the latent space 
            # representation of the input data (mu and sigma )'''

            x=self.Flatten.forward(x) #flatten the output of the last conv_layer before feeding to the mu and log_var dense layers

            mu=self.mu.forward(x)
            log_var=self.log_var.forward(x)

            return mu ,log_var

        
        def backward(self,mu_output_gradient,log_var_output_gradient,learning_rate):
            #to get the gradients of mu and var ,we have to combine the derivative of ELBO(reconstruction loss  and KL Loss) with respect to mu and log_var,
            #log_var is going to be used in the reparameterize class to compute sigma
            

            mu_gradient=self.mu.backward(mu_output_gradient,learning_rate)  #the output_gradient is the KL loss+ reconstruction loss for mu
            log_var_gradient=self.log_var.backward(log_var_output_gradient,learning_rate)

            #combine the gradientss of mu and log_var to get the gradients of the last conv layyer 
            output_gradient=mu_gradient+log_var_gradient #I got this from my manua derivations of the formuals on paper,
            #we have to add them togther because  the flatttened conv x output is the input of  the mu ad log_var dense layers(chain rule ect...) 
            # ,this helps us retain full information for our networ to learn better 

            output_gradient=self.Flatten.backward(output_gradient)

            for i in reversed(range(len(self.conv_layers))): #layer in reversed layers 
                input_gradient=self.conv_layers[i].activation.backward(output_gradient)#[layer.activation.backward()]  # i also have to differentiate with respect to RELu when moving backwards
                input_gradient =self.conv_layers[i].backward(input_gradient,learning_rate) # input gradent of the next layer coming from the output grdient og the previous layer when moving from the
                #last  conv layer to the --> first conv layer 
                output_gradient=input_gradient
            return input_gradient