# show forward diffusion process (mnist-fashion)

import torchvision
from torchvision.utils import save_image
import dill,sys,getopt

sys.path.append('../../src')
from DDM import *
from train_mnist_fashion import myload_mnist_fashion, reverse_transform

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy)

  return noisy_image


def Usage():
  print("Usage: python [opts] config_file output_directory")
  print("               -h  help")
  print("               -n  number of images")
  print("               -i  plot steps")
  print("")
  exit()
  

if __name__ == "__main__":
   
  plot_step = 1
  num_of_images = 12
  output_dir = "/tmp"

  try:
    opts, args = getopt.getopt(sys.argv[1:],"hn:i:")
  except getopt.GetoptError:
    Usage()
    
  if len(args) != 2:
    Usage()

  for opt, arg in opts:
    if opt == '-h':
      Usage()
    elif opt == '-n':
      num_of_images = int(arg)
    elif opt == '-i':
      plot_step = int(arg)

  config = dill.load(open(args[0], "rb"))
  print(f"loading config from {args[0]}")

  output_dir = args[1]

  ddm_parameters(config.timesteps)
  
  dataloader = myload_mnist_fashion(num_of_images)
  batch = next(iter(dataloader))  # get a random batch from training data
  batch = batch["pixel_values"] 

  images = []
  for i in range(0,num_of_images):
    img = reverse_transform(batch[i])
    images.append(img)
    save_image(img , f"{output_dir}/image_orig_i{i}.png")

    for t in range(0, config.timesteps, plot_step):
      img = get_noisy_image(batch[i], torch.tensor([t]))
      images.append(img)
      save_image(img , f"{output_dir}/image_t{t}_i{i}.png")  # un-normalized images

  frm = torchvision.utils.make_grid(images,nrow=config.timesteps//plot_step+1,scale_each=True, normalize=True, pad_value=1.0)

  save_image(frm, f"{output_dir}/images-summary.png")
  print(f"saving summary image to {output_dir}/images-summary.png")

