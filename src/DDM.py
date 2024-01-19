##### All  Modules for Diterministic Diffusion Model (DDM)  #####
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam

### various variance schedules

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def linear_alphas_bar_sqrt_schedule(timesteps):
  alphas_bar_sqrt = torch.linspace(0.99, 0.01, timesteps+2)
  return alphas_bar_sqrt[1:-1]**2

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
# x_start[Batch_Size, Channels, Width, Height]
# t[Batch_Size]
# noise[Batch_Size, Channels, Width, Height]
## return: x[Batch_Size, Channels, Width, Height]
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    # sqrt_alphas_cumprod[timesteps], sqrt_alphas_cumprod_t[Batch_Size,1,1,1]
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# denoise_model: U-Net model;
# x_start[Batch_Size, Channels, Width, Height]
# t[Batch_Size]
# noise[Batch_Size, Channels, Width, Height]
# loss_type: string
## return: loss as tensor
def p_losses(denoise_model, x_start, t, denoising_method, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    if denoising_method == 'ddm_x0':
      target = x_start
    elif denoising_method == 'ddm_noise':
      target = noise
    elif denoising_method == 'ddpm':
      target = noise
    else:
      raise NotImplementedError()

    predicted = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(target, predicted)
    elif loss_type == 'l2':
        loss = F.mse_loss(target, predicted)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target, predicted)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample_prob(model, x, t, t_index):
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    alphas_cumprod_t = extract(alphas_cumprod,t,x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    noise_pred = model(x, t)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

    x0_hat = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/alphas_cumprod_t

    if t_index != 0:
      posterior_variance_t = extract(posterior_variance, t, x.shape)
      noise = torch.randn_like(x)
      model_mean += torch.sqrt(posterior_variance_t) * noise

    return model_mean, x0_hat

@torch.no_grad()
def p_sample_ddm_w_noise(model, x, t, t_index):
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    alphas_cumprod_diff_t = extract(alphas_cumprod_diff,t,x.shape)
    alphas_cumprod_t = extract(alphas_cumprod,t,x.shape)

    # New update formulate in deterministic diffusion
    noise_pred = model(x, t)
    x0_hat = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/alphas_cumprod_t
    model_mean = sqrt_recip_alphas_t * (x - alphas_cumprod_diff_t * noise_pred)
    
    # if t_index != 0:
    #   posterior_variance_t = extract(posterior_variance, t, x.shape)
    #   noise = torch.randn_like(x)
    #   model_mean += 0.01*torch.sqrt(posterior_variance_t) * noise

    return model_mean, x0_hat

@torch.no_grad()
def p_sample_ddm_w_x0(model, x, t, t_index):
    gamma_ratio_t =  extract(gamma_ratio,t,x.shape)
    alphas_cumprod_t = extract(alphas_cumprod, t,x.shape)
    alphas_cumprod_prev_t = extract(alphas_cumprod_prev, t,x.shape)

    # New update formulate in deterministic diffusion
    x0_hat = model(x, t)
    model_mean = torch.sqrt(gamma_ratio_t)*x + (torch.sqrt(alphas_cumprod_prev_t) - torch.sqrt(gamma_ratio_t * alphas_cumprod_t))*x0_hat

    # if t_index != 0:
    #   posterior_variance_t = extract(posterior_variance, t, x.shape)
    #   noise = torch.randn_like(x)
    #   model_mean += 0.01*torch.sqrt(posterior_variance_t) * noise

    return model_mean, x0_hat

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cfg, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    x0_hat_all = []

    for i in reversed(range(0, cfg.timesteps)):
        if cfg.denoising_method == 'ddm_noise':
          [img, x0_hat] = p_sample_ddm_w_noise(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        elif cfg.denoising_method == 'ddm_x0':
          [img, x0_hat] = p_sample_ddm_w_x0(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        elif cfg.denoising_method == 'ddpm':
          [img, x0_hat] = p_sample_prob(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())
        x0_hat_all.append(x0_hat.cpu())

    return imgs, x0_hat_all

@torch.no_grad()
def sample(model, cfg, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, cfg, shape=(batch_size, channels, image_size, image_size))

class DDM_config():
    def __init__(self, device="cuda", epochs=10, timesteps=100, denoising_method='ddm_noise', \
                config_path="/tmp/config.pickle", image_size=28, channels=1):
        self.device = device
        self.epochs = epochs
        self.timesteps = timesteps
        self.denoising_method = denoising_method # ddm_noise / ddm_x0 / ddpm
        self.config_path = config_path
        self.image_size = image_size
        self.channels = channels
        
def train_model(model, dataloader, cfg):
  optimizer = Adam(model.parameters(), lr=1e-3)

  for epoch in range(cfg.epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch["pixel_values"].shape[0]
      batch = batch["pixel_values"].to(cfg.device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, cfg.timesteps, (batch_size,), device=cfg.device).long()

      ### loss = p_losses(model, batch, t, loss_type="huber")
      loss = p_losses(model, batch, t, cfg.denoising_method, loss_type="l2" )

      if step == 0:
        print(f"epoch {epoch}: Loss = {loss.item()}")

      loss.backward()
      optimizer.step()


def ddm_parameters(timesteps):
  global alphas_cumprod,alphas_cumprod_prev,alphas,betas, sqrt_recip_alphas
  global alphas_cumprod_diff,gamma_ratio, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance

  # # define beta schedule
  # betas = linear_beta_schedule(timesteps=timesteps)

  # define alphas
  # alphas = 1. - betas
  # alphas_cumprod = torch.cumprod(alphas, axis=0)
  # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
  # sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

  alphas_cumprod = linear_alphas_bar_sqrt_schedule(timesteps)
  alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
  alphas = alphas_cumprod /alphas_cumprod_prev
  betas = 1. - alphas
  sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

  # calculate for deterministic duffusion
  alphas_cumprod_diff = torch.sqrt(1. - alphas_cumprod) - torch.sqrt(alphas - alphas_cumprod)
  gamma_ratio = (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

  # calculations for diffusion q(x_t | x_{t-1}) and others
  sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
  sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

  # calculations for posterior q(x_{t-1} | x_t, x_0)
  posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
