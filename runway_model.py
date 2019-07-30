## Sample code to generate an image using the
## pre-trained PGAN celebAHQ-512 checkpoint from the Pytorch Hub

import torch
import runway
import numpy as np

@runway.setup(options={"checkpoint": runway.category(description="Pretrained checkpoints to use.",
                                      choices=['celebAHQ-512', 'celebAHQ-256', 'celeba'],
                                      default='celebAHQ-512')})
 def setup(opts):
   checkpoint = opts['checkpoint']
    use_gpu = True if torch.cuda.is_available() else False
    # Load the model from the Pytorch Hub
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                          'PGAN', model_name=checkpoint,
                           pretrained=True, useGPU=use_gpu)
   return model

use_gpu = True if torch.cuda.is_available() else False

# Generate one image
noise, _ = model.buildNoiseData(1)
with torch.no_grad():
    generated_image = model.test(noise)

# Now generated_image contains our generated image! 🌞
