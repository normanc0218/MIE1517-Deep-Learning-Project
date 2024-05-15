# Transfer learning to extract features using pre-trained VGG16 model
# Try it on Jupter notebook for a single frame
from torchvision import models
import torch.nn as nn
import math
import torch
from myutil import *
import time
from PIL import Image
import torchvision
from torchvision.transforms import functional as TVF
import torchvision.transforms as transforms


    
class VGG16FeatureExtractor(nn.Module):
  """
  Image feature extraction with MobileNet.
  """
  def __init__(self, reshape_size=224, pooling=False, verbose=False):
    super().__init__()

    self.vgg16 = models.vgg16(pretrained=True)
    # output of conv5_3 of vgg16 is N x 512 x 14 x 14
    self.vgg16 = nn.Sequential(*list(self.vgg16.features)[:30]) # layers up to and including the activation of conv5_3

    # adding a conv layer to make the output N x 1280 x 7 x 7
    self.conv = nn.Conv2d(in_channels=512, out_channels=1280, kernel_size=3, stride=2, padding=1)

    self.vgg16.add_module('LastConv', self.conv) # oitput is N x 1280 x 7 x 7

    # average pooling
    if pooling:
      self.vgg16.add_module('LastAvgPool', nn.AvgPool2d(math.ceil(reshape_size/32.))) # input: N x 512 x 14 x 14

    for i in self.vgg16.named_parameters():
      i[1].requires_grad = True # fine-tune all

    # if verbose:
    #   summary(self.vgg16.cuda(), (3, reshape_size, reshape_size))

  def forward(self, img, verbose=False):
    """
    Inputs:
    - img: Resized image, of shape 3x224x224

    Outputs:
    - feat: Image feature, of shape 1280 (pooled) or 1280x7x7
    """
    img_prepro = img.unsqueeze(0)  # Add batch dimension
    feat = self.vgg16(img_prepro).squeeze(-1).squeeze(-1)  # Forward and squeeze
    if verbose:
        print('Output feature shape: ', feat.shape)
    return feat
  
class RPN(nn.Module):
    def __init__(self):
        super().__init__()

        # READ ONLY
        self.anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
        #self.feat_extractor = FeatureExtractor()
        self.feat_extractor = VGG16FeatureExtractor()
        self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0]) #input size for mobile net 1280

    def load_weights(self, weights_file):
        # Load trained weights from the file
        trained_weights = torch.load(weights_file)

        # Extract the model's state dictionary from the loaded state dictionary
        model_state_dict = trained_weights['model_state_dict']

        # Print keys for the extracted model's state_dict and the model's state_dict
        # print("Keys in extracted model's state_dict:")
        # print(model_state_dict.keys())
        # print("\nKeys in model's state_dict:")
        # print(self.state_dict().keys())

        # Check if the extracted model's state_dict matches the model's architecture
        if set(model_state_dict.keys()) != set(self.state_dict().keys()):
            print("\nError: Model architecture mismatch. Loaded weights don't match model's architecture.")
            return

        # Assign the trained weights to the model parameters
        self.load_state_dict(model_state_dict)


    def inference(self, images, thresh=0.5, nms_thresh=0.7, mode='RPN'):

        assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'

        features, final_conf_probs, final_proposals = None, None, None

        # Here we predict the RPN proposal coordinates `final_proposals` and        #
        # confidence scores `final_conf_probs`.                                     #
        # The overall steps are similar to the forward pass but now you do not need  #
        # to decide the activated nor negative anchors.                              #
        # Threshold the conf_scores based on the threshold value `thresh`.     #
        # Then, apply NMS to the filtered proposals given the threshold `nms_thresh`.#


        final_conf_probs, final_proposals = [],[]
        # i) Image feature extraction
        features = self.feat_extractor(images)

        # ii) Grid and anchor generation
        grid = GenerateGrid()
        # anchors = GenerateAnchor(self.anchor_list.cuda(), grid)
        anchors = GenerateAnchor(self.anchor_list.to(grid.device, grid.dtype), grid)

        # iii) Compute conf_scores, proposals, class_prob through the prediction network
        conf_scores, offsets = self.prop_module(features)
        #offsets: (B, A, 4, H', W')
        #conf_scores: (B, A, 2, H', W')
        A,_,H,W = conf_scores.shape
        # Need to dig out from here 2024 03 11

        offsets = offsets.permute((0,2,3,1))
        # offsets=offsets.to(grid.device)
        # print(anchors.device)
        # print(offsets.device)

        proposals = GenerateProposal(anchors, offsets) #proposals:A,H,W,4
        # proposals is torch.Size([1, 13, 7, 7, 4])
        # transform
        # print(conf_scores.shape) #9 2 7 7 
        # print(proposals.shape) #9 7 7 4 

        conf_scores = torch.sigmoid(conf_scores[:,0,:,:]) # only look at the 1st confidence score which represent obj_conf
        conf_scores = conf_scores.permute((1,2,0)).reshape(-1)
        proposals = proposals.permute((1,2,0,3)).reshape(-1,4)
        # print(proposals,conf_scores)
        # Filter proposals by confidence scores
        mask1 = conf_scores > thresh
        sub_conf_scores = conf_scores[mask1]
        sub_proposals = proposals[mask1]


        # Apply NMS
        mask2 = nms(sub_proposals, sub_conf_scores, iou_threshold=nms_thresh)

        # Append filtered proposals and confidence scores
        final_proposals.append(sub_proposals[mask2,:])
        final_conf_probs.append(sub_conf_scores[mask2].unsqueeze(1))


        if mode == 'RPN':
            features = [torch.ones_like(i) for i in final_conf_probs] # dummy class
        
        return final_proposals, final_conf_probs, features

class TwoStageDetector(nn.Module):
  def __init__(self, in_dim=1280, hidden_dim=256, num_classes=4, \
               roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0)
    self.num_classes = num_classes # number of classes (excluding the background)
    self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
    self.rpn = RPN() # RPN model
    self.classificationLayer = nn.Sequential( # Define the classifier
          nn.Linear(in_dim,hidden_dim),
          nn.Dropout(drop_ratio),
          nn.ReLU(),
          nn.Linear(hidden_dim,self.num_classes)
        )
    self.MeanPool = nn.AvgPool2d((7,7))#added
  def load_weights(self, weights_file):
      # Load trained weights from the file
      trained_weights = torch.load(weights_file)

      # Extract the model's state dictionary from the loaded state dictionary
      model_state_dict = trained_weights['model_state_dict']

      # Print keys for the extracted model's state_dict and the model's state_dict
      # print("Keys in extracted model's state_dict:")
      # print(model_state_dict.keys())
      # print("\nKeys in model's state_dict:")
      # print(self.state_dict().keys())

      # Check if the extracted model's state_dict matches the model's architecture
      if set(model_state_dict.keys()) != set(self.state_dict().keys()):
          print("\nError: Model architecture mismatch. Loaded weights don't match model's architecture.")
          return

      # Assign the trained weights to the model parameters
      self.load_state_dict(model_state_dict)  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):

    final_proposals, final_conf_probs, final_class = None, None, None

    final_class=[]
    final_proposals, final_conf_probs, features = self.rpn.inference(images, thresh,
                                                                     nms_thresh,mode='FasterRCNN')

    aligned_features = torchvision.ops.roi_align(features, final_proposals,
                                                 (self.roi_output_w, self.roi_output_h))
    pooled_features = torch.mean(aligned_features,(2,3))
    cls_scores = self.classificationLayer(pooled_features)
    cls = torch.max(cls_scores,1)[1].to(torch.int64).unsqueeze(1)
    # slice cls into groups
    count = 0
    for i in range(len(final_proposals)):
      tmp_len=len(final_proposals[i])
      final_class.append(cls[count:count+tmp_len])
      count += tmp_len

    return final_proposals, final_conf_probs, final_class  
