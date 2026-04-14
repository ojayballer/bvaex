import os
from PIL import Image
import jax.numpy as jnp
class DataLoader:
    def __init__(self,batch_size=512,path="archive/img_align_celeba/img_align_celeba"):
        self.path=path
        self.batch_size=batch_size  
        self.image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        self.image_paths.sort()
    
    def load_batch(self,start):
        batch_paths=self.image_paths[start:start+self.batch_size]
        
        images=[]
        for batch in batch_paths:
             image= Image.open(batch)
             image=image.resize((64,64))
             image=jnp.array(image)/255  # normalize image to [0,1] because of I am using a sigmoid function
             image = jnp.transpose(image, (2, 0, 1))  
             images.append(image)
        return jnp.array(images)
        
    

    def __len__(self):
        return len(self.image_paths)//self.batch_size  # returns te total number of batches,a batch size  is 256


        