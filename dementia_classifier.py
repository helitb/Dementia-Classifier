"""
Dementia Classifier
Helit Bauberg 027466002
"""
import os
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler


def hello_nondemented_people():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from Classifer! Remember me?')
 
def show_what_you_doing(orig, new_sample):

  fig, axes = plt.subplots(1, 2, figsize=(10, 2))
  axes[0].imshow(orig)
  axes[0].set_title('Original Sample')
  axes[0].set_xticks([])  # Remove x-axis ticks
  axes[0].set_yticks([])  # Remove y-axis ticks

  axes[1].imshow(new_sample)
  axes[1].set_title('Generated Sample')
  axes[1].set_xticks([])  # Remove x-axis ticks
  axes[1].set_yticks([])  # Remove y-axis ticks

  plt.show()


# This takes care of generating more samples for undersampled classes
def AugmentClass(original_dataset, class_to_augment, 
    augmentation_transform, num_augmented_samples):
    
  print(len(original_dataset))
  im_count = 0

  # First, open all existing class samples from the original dataset as PIL
  for img, target in original_dataset.samples:
    if target == class_to_augment:
      im_count += 1
      imPIL = Image.open(img)      
      for _s in range(num_augmented_samples):
          augmented_img = augmentation_transform(imPIL)            
          new_name = img[0:-4] + '_New_' + str(_s) + '.jpg'
          augmented_img.save(new_name)
          #print(f"Writing {new_name} to directoy")  
          im_count+=1
      
       # every once in a while display the augmented image
      show_us = random.uniform(0.0,1.0)
      if (show_us < 0.0095):
        #print(show_us, im_count)
        show_what_you_doing(imPIL, augmented_img)

  return im_count

#####################################################################
# MODELS:   Model I (pytorch implementation of keras successful model):                                                #
#####################################################################

# input_size  = 3*176*176   # images 
# output_size = 4           # there are 4 classes

class ConvNet(nn.Module):
  
    def __init__(self, input_size=(3,176,176), num_classes=4, l_1=64):
      super(ConvNet, self).__init__()
      self.input_size = input_size
      self.output_size = num_classes
      self.l_1 = l_1

      # Feature map extraction
      self.feature_maps = {}

      # convolutional layers
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=l_1, kernel_size=5, padding=2) 
      self.conv2 = nn.Conv2d(in_channels=l_1, out_channels=(l_1*2), kernel_size=3, stride=1, padding=1)
      self.conv3 = nn.Conv2d(in_channels=(l_1*2), out_channels=(l_1*4), kernel_size=3, stride=1, padding=1)
      self.conv4 = nn.Conv2d(in_channels=(l_1*4), out_channels=(l_1*8), kernel_size=2, stride=1, padding=1) 
      self.conv5 = nn.Conv2d(in_channels=(l_1*8), out_channels=(l_1*16), kernel_size=2, stride=1, padding=1) 
      
      # max pooling layer
      self.pool = nn.MaxPool2d(2, 2)  
      
      # fully connected layers
      self.fc1 = nn.Linear((l_1*16) * 6 * 6, num_classes)
      #self.fc1 = nn.Linear((l_1*16) * 6 * 6, l_1*4)
      #self.fc2 = nn.Linear(l_1*4, num_classes)

      # dropout
      self.dropout1 = nn.Dropout(p=.25)
      self.dropout2 = nn.Dropout(p=.3)


    def forward(self, x):

     # feature maps captured for conv/activation/fc layers
     
      self.feature_maps['input'] = x
      x = F.relu(self.conv1(x.float()))
      self.feature_maps['conv1'] = x
      x = self.pool(x)
      self.feature_maps['pool_conv1'] = x

      x = F.relu(self.conv2(x))
      self.feature_maps['conv2'] = x
      x = self.pool(x)
      self.feature_maps['pool_conv2'] = x
      x = F.relu(self.conv3(x))
      self.feature_maps['conv3'] = x
      
      x = self.pool(x)
      self.feature_maps['pool_conv3'] = x
      x = self.dropout1(x)
      self.feature_maps['drop_conv3'] = x

      x = F.relu(self.conv4(x))
      self.feature_maps['conv4'] = x
      x = self.pool(x)
      self.feature_maps['pool_conv4'] = x
      x = F.relu(self.conv5(x))
      self.feature_maps['conv5'] = x
      x = self.pool(x)
      self.feature_maps['pool_conv5'] = x

      #x = self.dropout2(x)
      #print(x.size())

      # flattening
      x = x.flatten(start_dim=1)
      #print('after flatten', x.size()) #after flatten torch.Size([4, 1024])
       
      # fully connected layers
      x = self.fc1(x)
      #x = F.relu(self.fc1(x))
      #x = self.fc2(x)
      
      return x

    def extract_features(self):
      return self.feature_maps



#####################################################################
# MODELS:   Model II:  
#####################################################################
class ConvBlockNet(nn.Module):
    def __init__(self, input_size=(3,176,176), num_blocks=3, num_classes=4, l_1=64):
        super(ConvBlockNet, self).__init__()
      
        self.l_1 = l_1
        self.input_size = input_size
        self.in_channels = input_size[0]

        # First ConvBlock
        self.conv1_1 = nn.Conv2d(self.in_channels, l_1, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(l_1, l_1, kernel_size=3, padding=1)
        
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.3)
        
        # Second ConvBlock
        self.conv2_1 = nn.Conv2d(l_1, l_1*2, kernel_size=5, padding=1)
        self.conv2_2 = nn.Conv2d(l_1*2, l_1*2, kernel_size=5, padding=1)
        #self.conv2_3 = nn.Conv2d(l_1*2, l_1*2, kernel_size=3, padding=1)
        
        # Third ConvBlock
        self.conv3_1 = nn.Conv2d(l_1*2, l_1*4, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(l_1*4, l_1*4, kernel_size=3, padding=1)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(l_1*(2**(num_blocks-1)), l_1*4)
        self.fc2 = nn.Linear(l_1*4, l_1*2)
        self.fc3 = nn.Linear(l_1, num_classes)

    def forward(self, x):
      
        # First ConvBlock
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.downsample(x)
        #print('after conv1', x.size()) after conv1 torch.Size([4, 64, 86, 86])
        
        # Second ConvBlock
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        #x = self.relu(self.conv2_3(x))        
        x = self.downsample(x)
        #print('after conv2', x.size()) after conv2 torch.Size([4, 128, 43, 43])

        # Third ConvBlock
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        #print('after conv3', x.size())
        x = self.downsample(x)
        #print('after ds', x.size()) #after conv3 torch.Size([4, 256, 2, 2])
        
        # Global average pooling and classifier
        x = self.global_pool(x)
        #print('after pool', x.size())

        x = x.flatten(start_dim=1)
        #print('after flatten', x.size()) #after flatten torch.Size([4, 1024])
       
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
      
        return x



#####################################################################
# MODELS:   Baseline Model:  
#####################################################################
###### First Model  
class FirstConvNet(nn.Module):
  
    def __init__(self, input_size=(3,176,176), output_size=4, l_1=16):
      super(FirstConvNet, self).__init__()
      self.input_size = input_size
      self.output_size = output_size
      self.l1 = l_1

      # Feature map extraction
      self.feature_maps = {}

      # convolutional layer
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=l_1, kernel_size=5, padding=2) 
      self.conv2 = nn.Conv2d(l_1, l_1*2, 5, padding=2)
      self.conv3 = nn.Conv2d(l_1*2, l_1*4, 5, padding=2)
      
      # max pooling layer
      self.pool = nn.MaxPool2d(2, 2)  
      
      # fully connected layers
      self.fc1 = nn.Linear(l_1*4 * 5 * 5, l_1)
      self.fc2 = nn.Linear(l_1, self.output_size)
      
      # dropout
      self.dropout = nn.Dropout(p=.25)
      
    def forward(self, x):
        
        self.feature_maps['input'] = x

        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x.float()))
        self.feature_maps['conv1'] = x
        
        x = self.pool(x)
        self.feature_maps['pool_conv1'] = x

        x = F.relu(self.conv2(x))
        self.feature_maps['conv2'] = x
        
        x = self.pool(x)
        self.feature_maps['pool_conv2'] = x
        
        x = self.dropout(x)
        self.feature_maps['drop_conv2'] = x

        x = F.relu(self.conv3(x))
        self.feature_maps['conv3'] = x
        
        x = self.pool(x)
        self.feature_maps['pool_conv3'] = x
        print(x.shape)
      
        # flattening
        x = x.view(x.size(0),-1)
        #x = x.flatten(start_dim=1)
                
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
      
        return x

    def extract_features(self):
      return self.feature_maps


#####################################################################
# PART 2 - Train/Validate                                         #
#####################################################################

def get_train_val_loader(ds, batch_size, valid_size, loader_fraction):

  # obtain training indices that will be used for validation
  num_train = len(ds.targets)

  # loader_fraction = 1 for full data load, loader_fraction < 1 for smaller set
  num_of_indices = int(num_train * loader_fraction)

  all_indices = list(range(num_train))
  if loader_fraction < 1.0:
    all_indices = random.sample(all_indices, num_of_indices)
  else:
    np.random.shuffle(all_indices)
  
  #print(num_train,num_of_indices, len(all_indices))

  split = int(np.floor(valid_size * num_of_indices))
  train_idx, valid_idx = all_indices[split:], all_indices[:split]
  print('Splitting ', num_of_indices,'samples to: ' ,len(train_idx), 'training, ', len(valid_idx), 'validation')

  # define samplers for obtaining training and validation batches
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  # prepare data loaders 
  train_loader = DataLoader(ds, batch_size=batch_size, num_workers=2, sampler=train_sampler) #, pin_memory=True)
  valid_loader = DataLoader(ds, batch_size=batch_size, num_workers=2, sampler=valid_sampler) #, pin_memory=True)

  return train_loader, valid_loader






def train_model(mynet, full_ds, net_params):

  criterion = nn.CrossEntropyLoss()

  if net_params['optim'] == 'Adam':
    optimizer = optim.Adam(mynet.parameters(), 
                          lr=net_params['lr'], 
                          weight_decay=net_params['weight_decay'], 
                          )
  else:   # SGD   
    optimizer = optim.SGD(mynet.parameters(), 
                         lr=net_params['lr'], 
                         weight_decay=net_params['weight_decay'], 
                         momentum=net_params['momentum'])
  
  plt.rcParams['figure.figsize'] = (6.0, 4.0)
  plt.rcParams['font.size'] = 10

  epoch_t_loss = []
  epoch_v_loss = []
  v_accuracies = []
  
  # Get new train/validate dataset sampling on all of the data every epoch 
  t_loader, v_loader = get_train_val_loader(full_ds, net_params['batch_size'],
                      valid_size = 0.2, loader_fraction = 1.0)

  for epoch in range(net_params['epochs']):
    running_t_loss = 0.0
    t_loss = 0.0
    mb_t_loss = [] 

    print(f"Training - Epoch {epoch+1}")

    mynet.train(True)
    # Training loss
    for i, data in enumerate(t_loader,0):
      img, label = data
      img = torch.squeeze(img)
      #print(f"Input shape: {img.shape}, Label shape: {label.shape}")
      img = img.to(device='cuda')
      label = label.to(device='cuda')
      
      # zero the parameter gradients
      optimizer.zero_grad()
  
      # forward + backward + optimize
      outputs = mynet(img)
      t_batch_loss = criterion(outputs, label)
      t_batch_loss.backward()
      optimizer.step()

      # collect and print statistics
      t_loss += t_batch_loss.item()
      running_t_loss += t_batch_loss.item()

      if i % 50 == 49:    # log snapshot every 200 samples
        snapshot = running_t_loss / 50.0
        mb_t_loss.append(snapshot)         
        print("[%d, %5d] Training loss: %.3f" % (epoch + 1, i + 1, snapshot))
        running_t_loss = 0.0

    # When done with an epoch, plot training loss snapshots:
    plt.plot(mb_t_loss)

    print(f"Validation - Epoch {epoch+1}")

    mynet.eval()
    # Validation loss
    running_v_loss = 0.0
    v_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(v_loader, 0):
      with torch.no_grad():
        img, labels = data
        img = img.to(device='cuda')
        labels = labels.to(device='cuda')

        outputs = mynet(img)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        v_batch_loss = criterion(outputs, labels)
        v_loss += v_batch_loss.item()
        running_v_loss += v_batch_loss.item()

        if i % 50 == 49:    # log snapshot every 200 samples
          snapshot = running_v_loss / 50.0
          print("[%d, %5d] Validation loss: %.3f Accuracy: %2d%% (%2d/%2d)" % (epoch + 1, i + 1, snapshot,(100 * correct // total) , correct, total)) 
          running_v_loss = 0.0

    # Final loss values
    t_loss =  t_loss/len(t_loader)      
    v_loss =  v_loss/len(v_loader)      
    epoch_t_loss.append(t_loss)  
    epoch_v_loss.append(v_loss)  
    v_accuracies.append((100 * correct // total))
  
  ## printouts and visualisation of train/validation loss per epochs
  print('TRAIN/VAL END RESULTS: Achieved final Train Loss:', t_loss)
  print('TRAIN/VAL END RESULTS: Achieved final Validation Loss:', v_loss)
  print('TRAIN/VAL END RESULTS: Accuracy: %2d%% (%2d/%2d)' % ((100 * correct // total), correct, total))

  xlabel = "Loss Snapshots - " + str(epoch+1) + " Epochs"
  plt.xlabel(xlabel=xlabel)
  plt.ylabel("Training Loss")
  plt.show()  

  plt.plot(v_accuracies)
  xlabel = "Validation Accuracies Percentage / Epoch"
  plt.xlabel(xlabel=xlabel)
  plt.ylabel("Validation Accuracy")
  plt.show()  

  return mynet, epoch_t_loss, epoch_v_loss, outputs



#####################################################################
# Test and Evaluate:                                                   #
#####################################################################

def test_accuracy(model, loader, classes):
 
  correct_cls = [0. for i in range(classes)]
  total_cls = [0. for i in range(classes)]
  total = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  
  with torch.no_grad():
      for data in loader:
        img, label = data
        img = img.to(device='cuda')
        label = label.to(device='cuda')
        # calculate outputs by running images through the network
        outputs = model(img)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        
        predicted = predicted.to(device='cpu')
        predicted = predicted.detach().numpy()
        #print(predicted,predicted.shape) # (4,)

        label = label.to(device='cpu')
        label = label.detach().numpy()
        
        #outputs = outputs.to(device='cpu')
        #outputs = outputs.detach().numpy()
        #print(outputs, outputs.shape)     # (batch_size,classes)

        total += label.shape[0]
        correct += (predicted == label).sum().item()

        for i in range(outputs.shape[0]): ## for any batch size
          true_label = label.data[i]
          total_cls[true_label] += 1
          correct_cls[true_label] += (predicted.data[i] == true_label)

  accuracy =  (100 * correct // total)
  return accuracy,  correct_cls, total_cls







