### Sampling script to generate images ###

import torch
import torchvision
from torchvision.utils import save_image
import sys,getopt,dill

from UNet import *
from DDM import *

save_path = "/tmp"
plot_step = 1
num_of_images = 16

def Usage():
  print("Usage: python [opts] sampling.py config_file model")
  print("               -h  help")
  print("               -s  plot step")
  print("               -o  image save path")
  print("               -n  number of images to save")
  print("")
  exit()
  
try:
  opts, args = getopt.getopt(sys.argv[1:],"hs:o:n:")
except getopt.GetoptError:
  print("Usage: python [opts] sampling.py config_file model")
  exit()
if len(args) != 2:
  Usage()
for opt, arg in opts:
  if opt == '-h':
    Usage()
  elif opt == '-s':
    plot_step = int(arg)
  elif opt == '-o':
    save_path = arg
  elif opt == '-n':
    num_of_images = int(arg)

config = dill.load(open(args[0], "rb"))
print(f"loading config from {args[0]}")

model = torch.load(args[1])
print(f"loading model from {args[1]}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"using device: {device}{torch.cuda.current_device()}")

timesteps = config.timesteps
denoising_method = config.denoising_method 

ddm_parameters(config.timesteps)

# sample a batch of images
samples, x0_hats = sample(model, config, image_size=config.image_size, batch_size=num_of_images, channels=config.channels)

images = []
for i in range(0,num_of_images):
  for t in reversed(range(config.timesteps-1, -1, -plot_step)):
    images.append(samples[t][i])
    save_image((samples[t][i]+1)/2, f"{save_path}/sample_t{t}_i{i}.png") # un-normalized images  
  for t in reversed(range(config.timesteps-1, -1, -plot_step)):
    images.append(x0_hats[t][i])
    save_image((x0_hats[t][i]+1)/2 , f"{save_path}/x0-hat_t{t}_i{i}.png")  # un-normalized images
    
print(f"saving all generated image at {save_path}")
    
frm = torchvision.utils.make_grid(images,nrow=config.timesteps//plot_step,scale_each=True, normalize=True, pad_value=1.0)
save_image(frm,f"{save_path}/samples-summary.png")
print(f"saving a summary of all generated image at {save_path}/samples-summary.png")
