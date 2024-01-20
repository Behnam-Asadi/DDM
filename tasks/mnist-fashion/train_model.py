### Training Script for mnist-fashion ###

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Lambda  
from torchvision import transforms
import dill,sys,getopt
from datasets import load_dataset

sys.path.append('../../src')

from UNet import *
from DDM import *

# define image preprocessing transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
            transforms.Lambda(lambda t: (t * 2) - 1)
])

reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
#    Lambda(lambda t: t.clip(0,1)),
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]
   return examples

def myload_data(batch_size):
   dataset = load_dataset("fashion_mnist")

   # transform image pixels
   transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

   # create dataloader
   dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

   return dataloader

def Usage():
  print("Usage: python [opts] model_to_save")
  print("               -h  help")
  print("               -m  denoising method ('ddm_noise','ddm_x0','ddpm'")
  print("               -t  diffusion timmsteps")
  print("               -e  training epochs")
  print("               -b  training mini-batch size")
  print("               -c  config file to be saved")
  print("")
  exit()
  

if __name__ == "__main__":
   
   denoising_method = 'ddm_noise'
   timesteps = 15 
   epochs = 12
   batch_size = 128
   config_path = '/tmp/mnist_fashion.config'

   try:
      opts, args = getopt.getopt(sys.argv[1:],"hm:t:e:b:c:")
   except getopt.GetoptError:
       Usage()
   if len(args) != 1:
      Usage()

   for opt, arg in opts:
      if opt == '-h':
         Usage()
      elif opt == '-m':
         denoising_method = arg
      elif opt == '-t':
         timesteps = int(arg)
      elif opt == '-e':
         epochs = int(arg)
      elif opt == '-b':
         batch_size = int(arg)
      elif opt == '-c':
         config_path = arg

   model_path = args[0]

   print(f"timesteps={timesteps} epochs={epochs} batch_size={batch_size} denoising_method={denoising_method}")

   ### training starts HERE ...

   ddm_parameters(timesteps)

   dataloader = myload_data(batch_size)
   image_size = 28
   channels = 1

   device = "cuda" if torch.cuda.is_available() else "cpu"

   model = Unet(
      dim=image_size,
      channels=channels,
      dim_mults=(1, 2, 4,)
   )
   model.to(device)
   print(f"using device: {device}{torch.cuda.current_device()}")

   config = DDM_config(device, epochs=epochs, timesteps=timesteps, denoising_method=denoising_method, image_size=image_size, \
                       channels=channels, config_path=config_path)

   train_model(model, dataloader, config)

   # save the model
   torch.save(model, model_path)
   print(f"saving model as {model_path}")

   # Save the config file
   dill.dump(config, file = open(config.config_path, "wb"))
   print(f"saving config as {config.config_path}")
