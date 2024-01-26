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
save_all_images = False 
cuda_id = 0
block_size = 100

def Usage():
  print("Usage: python [opts] sampling.py config_file model")
  print("               -h  help")
  print("               -s  plot step")
  print("               -o  image save path")
  print("               -n  number of images to save")
  print("               -b  block size")
  print("               -m  save intermediate images at all timesteps")
  print("               -g  choose gpu id: 0/1/2/3... (-1: use all gpus)")
  print("")
  exit()
  
try:
  opts, args = getopt.getopt(sys.argv[1:],"hs:o:n:mg:b:")
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
  elif opt == '-m':
    save_all_images = True
  elif opt == '-g':
    cuda_id = int(arg)
  elif opt == '-b':
    block_size = int(arg)

if (num_of_images<block_size):
  block_size = num_of_images
num_of_images = (num_of_images // block_size) * block_size

print(f"generating {num_of_images} images")
    
config = dill.load(open(args[0], "rb"))
print(f"loading config from {args[0]}")

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
  if cuda_id >= 0:
    device = f"cuda:{cuda_id}"
  else:
    device = "cuda"
else:
  device = "cpu"

model = torch.load(args[1],map_location=device)
print(f"loading model from {args[1]}")

model.to(device)
if cuda_id < 0:
  model = nn.DataParallel(model)   ### use all GPUs
  print(f"using device: all {device}s")
else:
  print(f"using device: {device}")

timesteps = config.timesteps
denoising_method = config.denoising_method 

ddm_parameters(config.timesteps)

# sampling images
samples = []
x0_hats = []
for b in range(0,num_of_images//block_size):
  sam, x0 = sample(model, config, image_size=config.image_size, batch_size=block_size, channels=config.channels)  
  samples.append(sam)
  x0_hats.append(x0)

images = []
if save_all_images == False:
  for i in range(0,num_of_images):
    images.append(x0_hats[i//block_size][config.timesteps-1][i%block_size])
    save_image((x0_hats[i//block_size][config.timesteps-1][i%block_size]+1)/2 , f"{save_path}/x0-hat_t{config.timesteps-1}_i{i}.png")
                                             # un-normalized images
  print(f"saving all final images x0 into {save_path}")
else:
  for i in range(0,num_of_images):
    for t in reversed(range(config.timesteps-1, -1, -plot_step)):
      images.append(samples[i//block_size][t][i%block_size])
      save_image((samples[i//block_size][t][i%block_size]+1)/2, f"{save_path}/sample_t{t}_i{i}.png") # un-normalized images  
    for t in reversed(range(config.timesteps-1, -1, -plot_step)):
      images.append(x0_hats[i//block_size][t][i%block_size])
      save_image((x0_hats[i//block_size][t][i%block_size]+1)/2 , f"{save_path}/x0-hat_t{t}_i{i}.png")  # un-normalized images
  print(f"saving all images generated at all timesteps into {save_path}")
  frm = torchvision.utils.make_grid(images,nrow=config.timesteps//plot_step,scale_each=True, normalize=True, pad_value=1.0)
  save_image(frm,f"{save_path}/samples-summary.png")
  print(f"saving a summary of all generated image at {save_path}/samples-summary.png")
