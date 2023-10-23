import torch
import torch.nn as nn
import numpy as np
from Layers import layers
from torch.nn import functional as F


def fc(input_shape, num_classes, dense_classifier=False, pretrained=False, L=4, N=100, nonlinearity=nn.ReLU(), init_type='uniform'):
  size = np.prod(input_shape)
  # hidden_dims = [100, 100, 100, 100, 100]
  # Linear feature extractor
  modules = [nn.Flatten()]
  modules.append(layers.Linear(size, N))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Linear(N,N))
    modules.append(nonlinearity)

  # Linear classifier
  if dense_classifier:
    modules.append(nn.Linear(N, num_classes))
  else:
    modules.append(layers.Linear(N, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  def _initialize_weights(model, init_type):
      if 'kaiming' in init_type:
        for m in model.modules():
            if isinstance(m, (layers.Linear, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif 'xavier' in init_type:
        for m in model.modules():
            if isinstance(m, (layers.Linear, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
      else:
        pass

  if init_type is not 'uniform':
    _initialize_weights(model, init_type)

  return model


def conv(input_shape, num_classes, dense_classifier=False, pretrained=False, L=3, N=32, nonlinearity=nn.ReLU()): 
  channels, width, height = input_shape
  
  # Convolutional feature extractor
  modules = []
  modules.append(layers.Conv2d(channels, N, kernel_size=3, padding=3//2))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Conv2d(N, N, kernel_size=3, padding=3//2))
    modules.append(nonlinearity)
      
  # Linear classifier
  modules.append(nn.Flatten())
  if dense_classifier:
    modules.append(nn.Linear(N * width * height, num_classes))
  else:
    modules.append(layers.Linear(N * width * height, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  return model