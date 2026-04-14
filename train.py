from load_data import DataLoader 
from model.model import CONVAE
import time
import os
import jax 
import jax.numpy as jnp
class Train:
    def __init__(self,input,input_shape,kernel_size,encoder_filters,decoder_filters,latent_dim,learning_rate):
        #this learning t\rate is not doing anything again ,since I am now using adam optimizer
        self.learning_rate=learning_rate
        self.model=CONVAE(input,input_shape,kernel_size,encoder_filters,decoder_filters,latent_dim)
        self.dataloader=DataLoader()
        self.learning_rate=learning_rate
        self.key = jax.random.PRNGKey(42)

    

    def train(self,epochs):
        for i in range(epochs):
            start_time=time.time()
            print(f"Starting Training for Epoch {i+1}/{epochs}")
            losses=[] ;rl_losses=[] ; kl_losses=[]
            for  start in range(0,len(self.dataloader)*self.dataloader.batch_size,self.dataloader.batch_size) : # start ,stop ,step ->stop represnts the no of batches of size batch_size
                              #we will ignore the last batch if it does not form the  complete 32 images 
                batch=self.dataloader.load_batch(start) # i is the start index of the batch 
                self.key,key_b=jax.random.split(self.key)
                kl_weight =0.5 #no kl-annealing,B-VAE
                #kl_weight = min(0.1, (i * len(self.dataloader) + (start // self.dataloader.batch_size)) / (20 * len(self.dataloader))) #kl-annealing,kl-weight per batch 
                loss,rl_loss,kl_loss=self.model.forward(batch,key_b,kl_weight)
                losses.append(loss)  ; rl_losses.append(rl_loss) ; kl_losses.append(kl_loss)
                self.model.adam.step()
                self.model.backward(self.learning_rate)
                print(f"Epoch {i + 1}/{epochs}, Batch: {start // self.dataloader.batch_size}/{len(self.dataloader)}, Loss: {loss:.4f}, RL: {rl_loss:.4f}, KL: {kl_loss:.4f}, Time: {time.time() - start_time:.2f}s")
            self.save_model(self.model, path=f"saved_weights/epoch_{i+1}")
        return losses,rl_losses,kl_losses


    def save_model(self,model,path="saved weights"):
         os.makedirs(path, exist_ok=True)
         
         #Encoder saved weights
         for i,layer in enumerate(self.model.encoder.conv_layers):
             jnp.save(f"{path}/encoder_conv_{i}_weights",layer.weights) # save each encoders conv weight
             jnp.save(f"{path}/encoder_conv_{i}_biases",layer.biases)   #save each encoder's conv biases

         #save encoder's dense layer mu weights
         jnp.save(f"{path}/mu_s_weights",self.model.encoder.mu.weight) # save mu's weight
         jnp.save(f"{path}/mu_s_biases",self.model.encoder.mu.bias) #save mu's bias


         #save encoder's dense layer log_var weights
         jnp.save(f"{path}/log_var_s_weight",self.model.encoder.log_var.weight) #weight
         jnp.save(f"{path}/log_var_s_bias",self.model.encoder.log_var.bias) #bias

        
         #save decoders dense layer
         jnp.save(f"{path}/decoder_s_dense_layer_weight",self.model.decoder.dense_layer.weight) #weight
         jnp.save(f"{path}/decoder_s_dense_layer_bias",self.model.decoder.dense_layer.bias)#bias

         #save decoder's weight
         for i ,layer in enumerate(self.model.decoder.transposed_conv_layers):
             jnp.save(f"{path}/decoder_transposed_conv_{i}_weights",layer.weights)   #save each decoder transposed conv weight
             jnp.save(f"{path}/decoder_transposed_conv_{i}_biases",layer.biases)   #save each decoders transposed conv bias
             


def main():
      
      lets_train =Train(input=(3, 64, 64),input_shape=(128, 8,8),kernel_size=4,encoder_filters=[32, 64, 128],decoder_filters=[64,32,3],latent_dim=128,learning_rate=0.0001)
      lets_train.train(100) # 100 epochs
      lets_train.save_model(lets_train.model)
    
if __name__ == "__main__":
    main()