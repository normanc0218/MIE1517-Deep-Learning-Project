import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn as nn
import math

class ProposalModule(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
        super().__init__()

        assert num_anchors != 0
        self.num_anchors = num_anchors

        self.predictHead = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.Dropout(drop_ratio),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, 6 * self.num_anchors, 1)
        )

    def forward(self, features):
        anchor_features = self.predictHead(features)  # ((A*6) x H x W)
        # print(anchor_features.shape) #1 54 7 7 
        _,_, H, W = anchor_features.shape
        anchor_features = anchor_features.reshape(self.num_anchors,6,H,W)
        # Split features into conf_package and offsets_package
        conf_package = anchor_features[:, :2, :, :]  # (A x 2 x H x W)
        offsets_package = anchor_features[:, 2:, :, :]  # (A x 4 x H x W)

        # Return confidence scores and offsets
        return conf_package, offsets_package
    
class VGG16FeatureExtractor(nn.Module):

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
        - img: Single resized image, of shape 3x224x224

        Outputs:
        - feat: Image feature, of shape 1280 (pooled) or 1280x7x7
        """
        # Pass the single image through the VGG16 model
        feat = self.vgg16(img.unsqueeze(0)).squeeze(-1).squeeze(-1)  # Forward and squeeze

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


def coord_trans(bbox, w_pixel, h_pixel, w_amap=7, h_amap=7, mode='a2p'):
  #  w_amap=7, h_amap=7 will follow the backbone feature map
  assert mode in ('p2a', 'a2p'), 'invalid coordinate transformation mode!'
  assert bbox.shape[-1] >= 4, 'the transformation is applied to the first 4 values of dim -1'

  if bbox.shape[0] == 0: # corner cases
    print("it is shape 0")
    return bbox

  resized_bbox = bbox.detach().clone() # detach for requre grad error
  # in that case, w_pixel and h_pixel will be scalars
  resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
  invalid_bbox_mask = (resized_bbox == -1) # indicating invalid bbox

  if mode == 'p2a':
    # pixel to activation
    width_ratio = w_pixel * 1. / w_amap
    height_ratio = h_pixel * 1. / h_amap
    resized_bbox[:, :, [0, 2]] /= width_ratio
    resized_bbox[:, :, [1, 3]] /= height_ratio
  else:
    # activation to pixel
    width_ratio = w_pixel * 1. / w_amap
    height_ratio = h_pixel * 1. / h_amap
    print(w_pixel,h_pixel)
    resized_bbox[:, :, [0, 2]] *= width_ratio
    resized_bbox[:, :, [1, 3]] *= height_ratio

  resized_bbox.masked_fill_(invalid_bbox_mask, -1)
  resized_bbox.resize_as_(bbox)
  return resized_bbox


def ConfScoreRegression(conf_scores):

  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1.
  GT_conf_scores[M:, 1] = 1.

  conf_score_loss = F.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum')
  return conf_score_loss

def BboxRegression(offsets, GT_offsets):

  bbox_reg_loss = F.smooth_l1_loss(offsets, GT_offsets, reduction='sum') 
  return bbox_reg_loss


def IoU(proposals, bboxes):
    """
    Compute Intersection over Union (IoU) between proposals and ground truth bounding boxes for one frame.

    Inputs:
    - proposals: Tensor of shape (A, H, W, 4) containing proposal coordinates
    - bboxes: Tensor of shape (N, 4) containing ground truth bounding boxes

    Outputs:
    - iou_mat: Tensor of shape (A*H*W, N) containing IoU values for each proposal and ground truth bbox pair
    """
    A, H, W, _ = proposals.shape
    proposals = proposals.view(A*H*W, 4)

    # Compute top-left and bottom-right coordinates of intersection
    tl = torch.max(proposals[:, :2].unsqueeze(1), bboxes[:, :2].unsqueeze(0))
    br = torch.min(proposals[:, 2:].unsqueeze(1), bboxes[:, 2:4].unsqueeze(0))

    # Compute intersection area
    intersect = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)

    # Compute area of bounding boxes and proposals
    a = torch.prod(bboxes[:, 2:4] - bboxes[:, :2], dim=1)
    b = torch.prod(proposals[:, 2:] - proposals[:, :2], dim=1)

    # Compute IoU
    iou_mat = torch.div(intersect, a.unsqueeze(0) + b.unsqueeze(1) - intersect)

    return iou_mat



def GenerateGrid(w_amap=7, h_amap=7, dtype=torch.float32, device='cpu'):
    # Generate horizontal and vertical ranges representing the centers of grid cells
    w_range = torch.arange(0, w_amap, dtype=dtype, device=device) + 0.5
    h_range = torch.arange(0, h_amap, dtype=dtype, device=device) + 0.5

    # Generate grid indices by repeating and stacking horizontal and vertical ranges
    w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
    h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
    grid = torch.stack([w_grid_idx, h_grid_idx], dim=-1)

    return grid




def GenerateAnchor(anc, grid):
    anchors = None
    H, W, _ = grid.shape
    A, _ = anc.shape
    anchors = torch.zeros((A, H, W, 4), device = grid.device, dtype = grid.dtype)
    for a in range(A):
      anchors[a,:,:,0] = grid[:,:,0] - anc[a,0]/2
      anchors[a,:,:,1] = grid[:,:,1] - anc[a,1]/2
      anchors[a,:,:,2] = grid[:,:,0] + anc[a,0]/2
      anchors[a,:,:,3] = grid[:,:,1] + anc[a,1]/2

    return anchors


def GenerateProposal(anchors, offsets):
  proposals = None
  proposals = torch.zeros_like(anchors)
  anc_trans = torch.zeros_like(anchors)

  anc_trans[:, :, :, 2:] = (anchors[:, :, :, 2:] - anchors[:, :, :, :2]) # w, h = br - tl
  anc_trans[:, :, :, :2] = (anchors[:, :, :, 2:] + anchors[:, :, :, :2]) / 2 # (br + tl) / 2
  new_anc_trans = anc_trans.clone() # avoid inplace operation
  new_anc_trans[:, :, :, :2] = anc_trans[:, :, :, :2] + offsets[:, :, :, :2] * anc_trans[:, :, :, 2:]
  new_anc_trans[:, :, :, 2:] = torch.mul(anc_trans[:, :, :, 2:], torch.exp(offsets[:, :, :, 2:]))

  # tansform back
  proposals[:, :, :, :2] =  new_anc_trans[:, :, :, :2] - (new_anc_trans[:, :, :, 2:] / 2)
  proposals[:, :, :, 2:] =  new_anc_trans[:, :, :, :2] + (new_anc_trans[:, :, :, 2:] / 2)
  # print("From 1")
  return proposals

def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None

  keep = []
  # print(keep.dtype)
  indexing = torch.argsort(scores, descending=True)
  boxes_sort = boxes[indexing, :]
  # print(boxes_sort)
  areas = torch.prod(boxes[:, 2:] - boxes[:, :2], dim=1)
  # print(areas.shape)
  while indexing.size()[0] > 0:
    # still left
    # print(indexing.size()[0])
    idx = indexing[0]
    max_box = boxes[idx] # current max
    # print(keep)
    # print(idx)
    #torch.cat((keep, idx))
    keep.append(idx)
    # compute iou:
    tl = torch.max(max_box[:2], boxes[indexing][:, :2]) # should broadcast
    # print("tl is", tl)
    br = torch.min(max_box[2:], boxes[indexing][:, 2:])
    #print(torch.prod(br - tl, dim=3))
    intersect = torch.prod(br - tl, dim=1) * (tl < br).all(dim=1)
    # print(intersect.shape)
    a = areas[idx] # (1, )
    b = areas #(N, 1)

    iou_mat = torch.div(intersect, a + b[indexing] - intersect).squeeze() #(N, )
    # print(iou_mat)
    left = torch.where(iou_mat <= iou_threshold)
    indexing = indexing[left]
    # print(indexing.shape)
    # print(left)
  if topk is None:
    pass
  else:
    keep = keep[:topk]
  keep = torch.tensor(keep, **{'dtype': torch.long, 'device': 'cpu'}).to(scores.device)
  return keep

def data_visualizer(img, idx_to_class, bbox=None, pred=None):
    img_copy = np.array(img)

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4].int()  # Ensure integer type
            cv2.rectangle(img_copy, (int(one_bbox[0]), int(one_bbox[1])), (int(one_bbox[2]), int(one_bbox[3])), (255, 0, 0), 2)
            if bbox.shape[1] > 4:  # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                cv2.putText(img_copy, '%s' % obj_cls, (int(one_bbox[0]), int(one_bbox[1])+15), cv2.FONT_HERSHEY_PLAIN, 100.0, (0, 0, 255), thickness=10)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4].int()  # Ensure integer type
            cv2.rectangle(img_copy, (int(one_bbox[0]), int(one_bbox[1])), (int(one_bbox[2]), int(one_bbox[3])), (0, 255, 0), 2)

            if pred.shape[1] > 4:  # if class and conf score info provided
                # print(pred[bbox_idx][4].item())
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score), (int(one_bbox[0]), int(one_bbox[1])+15), cv2.FONT_HERSHEY_PLAIN, 100.0, (255, 0, 255), thickness=10)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()