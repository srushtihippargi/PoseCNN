import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
from torchvision.ops import RoIPool

import numpy as np
import random
import statistics
import time
from typing import Dict, List, Callable, Optional

from rob599 import quaternion_to_matrix
from p3_helper import HoughVoting, _LABEL2MASK_THRESHOL, loss_cross_entropy, loss_Rotation, IOUselection


_HOUGHVOTING_NUM_INLIER = 500
_HOUGHVOTING_DIRECTION_INLIER = 0.9


def HoughVoting(label, centermap, num_classes=10):
    """
    label [bs, 3, H, W]
    centermap [bs, 3*maxinstance, H, W]
    """

    batches, H, W = label.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)

    xv, yv = np.meshgrid(x, y)

    xy = torch.from_numpy(np.array((xv, yv))).to(device = label.device, dtype=torch.float32)
    x_index = torch.from_numpy(x).to(device = label.device, dtype=torch.int32)

    centers = torch.zeros(batches, num_classes, 2)
    depths = torch.zeros(batches, num_classes)

    for bs in range(batches):
        for cls in range(1, num_classes + 1):
            if (label[bs] == cls).sum() >= _LABEL2MASK_THRESHOL:
                pixel_location = xy[:2, label[bs] == cls]
                pixel_direction = centermap[bs, (cls-1)*3:cls*3][:2, label[bs] == cls]
                
                y_index = x_index.unsqueeze(dim=0) - pixel_location[0].unsqueeze(dim=1)
                y_index = torch.round(pixel_location[1].unsqueeze(dim=1) + (pixel_direction[1]/pixel_direction[0]).unsqueeze(dim=1) * y_index).to(torch.int32)
                
                mask = (y_index >= 0) * (y_index < H)
                count = y_index * W + x_index.unsqueeze(dim=0)

                center, inlier_num = torch.bincount(count[mask]).argmax(), torch.bincount(count[mask]).max()
                center_x, center_y = center % W, torch.div(center, W, rounding_mode='trunc')

                if inlier_num > _HOUGHVOTING_NUM_INLIER:
                    centers[bs, cls - 1, 0], centers[bs, cls - 1, 1] = center_x, center_y
                    
                    xyplane_dis = xy - torch.tensor([center_x, center_y])[:, None, None].to(device = label.device)
                    xyplane_direction = xyplane_dis/(xyplane_dis**2).sum(dim=0).sqrt()[None, :, :]

                    predict_direction = centermap[bs, (cls-1)*3:cls*3][:2]
                    inlier_mask = ((xyplane_direction * predict_direction).sum(dim=0).abs() >= _HOUGHVOTING_DIRECTION_INLIER) * label[bs] == cls
                    
                    depths[bs, cls - 1] = centermap[bs, (cls-1)*3:cls*3][2, inlier_mask].mean()
    return centers, depths


def hello_pose_cnn():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from pose_cnn.py!")


class FeatureExtraction(nn.Module):
    """
    Feature Embedding Module for PoseCNN. Using pretrained VGG16 network as backbone.
    """    

    def __init__(self, pretrained_model):
        super(FeatureExtraction, self).__init__()
        embedding_layers = list(pretrained_model.features)[:30]
        self.embedding1 = nn.Sequential(*embedding_layers[:23])
        self.embedding2 = nn.Sequential(*embedding_layers[23:])

        for i in [0, 2, 5, 7, 10, 12, 14]:
            self.embedding1[i].weight.requires_grad = False
            self.embedding1[i].bias.requires_grad = False
    

    def forward(self, datadict):
        """
        feature1: [bs, 512, H/8, W/8]
        feature2: [bs, 512, H/16, W/16]
        """ 

        feature1 = self.embedding1(datadict['rgb'])
        feature2 = self.embedding2(feature1)
        return feature1, feature2


class SegmentationBranch(nn.Module):
    """
    Instance Segmentation Module for PoseCNN. 
    """    

    def __init__(self, num_classes = 10, hidden_layer_dim = 64):
        super(SegmentationBranch, self).__init__()

        self.num_classes = num_classes
        self.convf1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1, stride=1, padding=0)
        self.convf2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1, stride=1, padding=0)
        
        torch.nn.init.kaiming_normal_(self.convf1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.convf2.weight, mode='fan_out', nonlinearity='relu')
        
        torch.nn.init.zeros_(self.convf1.bias)
        torch.nn.init.zeros_(self.convf2.bias)

        self.conv_f_res = nn.Conv2d(hidden_layer_dim, self.num_classes+1, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.conv_f_res.weight, mode='fan_out', nonlinearity='relu')

        torch.nn.init.zeros_(self.conv_f_res.bias)


    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            probability: Segmentation map of probability for each class at each pixel.
                probability size: (B,num_classes+1,H,W)
            segmentation: Segmentation map of class id's with highest prob at each pixel.
                segmentation size: (B,H,W)
            bbx: Bounding boxs detected from the segmentation. Can be extracted 
                from the predicted segmentation map using self.label2bbx(segmentation).
                bbx size: (N,6) with (batch_ids, x1, y1, x2, y2, cls)
        """

        probability = None
        segmentation = None
        bbx = None
        
        self.feature1 = feature1
        self.feature2 = feature2

        feature_1 = self.convf1(feature1)
        feature_2 = self.convf2(feature2)

        feature_1 = torch.nn.functional.relu(feature_1)
        feature_2 = torch.nn.functional.relu(feature_2)
        
        f2_upsample = torch.nn.functional.interpolate(feature_2, size=feature_1.shape[2:], 
                                                       mode='bilinear')
        f_final = feature_1 + f2_upsample

        f_final = self.conv_f_res(f_final)
        soft_max = torch.nn.Softmax(dim=1)

        probability = soft_max(f_final)
        probability = torch.nn.functional.interpolate(probability, 
                                                      size=(480,640), 
                                                      mode='bilinear', 
                                                      align_corners = True)
          
        segmentation = torch.argmax(probability, dim=1)
        bbx = self.label2bbx(segmentation)

        return probability, segmentation, bbx
    

    def label2bbx(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(1, self.num_classes, 1, 1).to(device)
        label_target = torch.linspace(0, self.num_classes - 1, steps = self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)

        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0: 
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    if y.numel() >= _LABEL2MASK_THRESHOL: bbx.append([batch_id, 
                                                                      torch.min(x).item(), 
                                                                      torch.min(y).item(), 
                                                                      torch.max(x).item(), 
                                                                      torch.max(y).item(), 
                                                                      cls_id])
                        
        bbx = torch.tensor(bbx).to(device)
        return bbx
        
        
class TranslationBranch(nn.Module):
    """
    3D Translation Estimation Module for PoseCNN. 
    """    

    def __init__(self, num_classes = 10, hidden_layer_dim = 128):
        super(TranslationBranch, self).__init__()
        
        self.num_classes = num_classes
        self.convf1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1, stride=1, padding=0)
        self.convf2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1, stride=1, padding=0)
        
        torch.nn.init.kaiming_normal_(self.convf1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.convf2.weight, mode='fan_out', nonlinearity='relu')
        
        torch.nn.init.zeros_(self.convf1.bias)
        torch.nn.init.zeros_(self.convf2.bias)
        
        self.conv_f_res = nn.Conv2d(hidden_layer_dim, 3*self.num_classes, kernel_size=1) 
        torch.nn.init.kaiming_normal_(self.conv_f_res.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.conv_f_res.bias)


    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            translation: Map of object centroid predictions.
                translation size: (N,3*num_classes,H,W)
        """
        
        translation = None

        feat_1 = self.convf1(feature1)
        feat_2 = self.convf2(feature2)

        feat_1 = torch.nn.functional.relu(feat_1)
        feat_2 = torch.nn.functional.relu(feat_2)

        f2_upsample = torch.nn.functional.interpolate(feat_2, size=feat_1.shape[2:], mode='bilinear')
        feat_final = feat_1 + f2_upsample

        feat_final = self.conv_f_res(feat_final)
        translation = torch.nn.functional.interpolate(feat_final, 
                                                      size=(480,640), 
                                                      mode='bilinear', 
                                                      align_corners = True)

        return translation


class RotationBranch(nn.Module):
    """
    3D Rotation Regression Module for PoseCNN. 
    """   

    def __init__(self, feature_dim = 512, roi_shape = 7, hidden_dim = 4096, num_classes = 10):
        super(RotationBranch, self).__init__()

        self.num_classes = num_classes

        self.roi_pool1 = RoIPool(output_size=roi_shape, spatial_scale=1/8) 
        self.roi_pool2 = RoIPool(output_size=roi_shape, spatial_scale=1/16) 
        
        self.fc1 = nn.Linear(roi_shape*roi_shape*512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4*num_classes)
        
        kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.fc1.bias.data.zero_()

        kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.fc2.bias.data.zero_()

        kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        self.fc3.bias.data.zero_()


    def forward(self, feature1, feature2, bbx):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
            bbx: Bounding boxes of regions of interst (N, 5) with (batch_ids, x1, y1, x2, y2)
        Returns:
            quaternion: Regressed components of a quaternion for each class at each ROI.
                quaternion size: (N,4*num_classes)
        """
        
        quaternion = None

        bbx = bbx.float()
        feat_1 = self.roi_pool1(feature1, bbx)
        feat_2 = self.roi_pool2(feature2, bbx)
        
        feat_final = feat_1 + feat_2
        feat_final = feat_final.view(feat_final.shape[0], -1)
    
        out_fc1 = self.fc1(feat_final)
        out_fc2 = self.fc2(out_fc1)
        out_fc3 = self.fc3(out_fc2)

        quaternion = out_fc3

        return quaternion


class PoseCNN(nn.Module):
    """
    PoseCNN
    """

    def __init__(self, pretrained_backbone, models_pcd, cam_intrinsic):
        super(PoseCNN, self).__init__()

        self.iou_threshold = 0.7
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic

        self.feature_extraction = FeatureExtraction(pretrained_model=pretrained_backbone).to(device = models_pcd.device)
        self.segmentation_branch= SegmentationBranch().to(device=models_pcd.device)

        self.translation_branch = TranslationBranch().to(device=models_pcd.device)
        self.rotation_branch = RotationBranch().to(device=models_pcd.device)


    def forward(self, input_dict):
        """
        input_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs'
        }
        """

        if self.training:
            loss_dict = {
                "loss_segmentation": 0,
                "loss_centermap": 0,
                "loss_R": 0
            }

            gt_bbx = self.getGTbbx(input_dict)
            label = input_dict['label']
            
            feat_1, feat_2 = self.feature_extraction(input_dict)
            probability, segmentation, bbx_pred = self.segmentation_branch(feat_1, feat_2)
            loss_dict["loss_segmentation"] = loss_cross_entropy(probability, label)
            
            translation_map = self.translation_branch(feat_1, feat_2)
            loss_dict['loss_centermap'] = nn.functional.l1_loss(translation_map, input_dict['centermaps'])
            filtered_bbx = IOUselection(bbx_pred, gt_bbx, threshold=self.iou_threshold)

            if filtered_bbx.shape[0] > 0:
              quaternion_map = self.rotation_branch(feat_1, feat_2, filtered_bbx[:, :-1])
              pred_rotation, pred_labels = self.estimateRotation(quaternion_map, filtered_bbx)
              
              gt_rotation = self.gtRotation(filtered_bbx, input_dict)
              loss_dict["loss_R"] = loss_Rotation(pred_rotation, gt_rotation, pred_labels, self.models_pcd)

            else: loss_dict["loss_R"] = torch.tensor(0.0, device=bbx_pred.device)

        else:
            output_dict = None
            segmentation = None

            with torch.no_grad():
                feat_1, feat_2 = self.feature_extraction(input_dict)
                probability, segmentation, bbx_pred = self.segmentation_branch(feat_1, feat_2)
                
                translation_map = self.translation_branch(feat_1,feat_2)
                pred_centers, pred_depths = HoughVoting(segmentation, translation_map, num_classes=10)
        
                quaternion_map = self.rotation_branch(feat_1, feat_2, bbx_pred[:, :-1])
                pred_rotation, pred_labels = self.estimateRotation(quaternion_map, bbx_pred)
                
                output_dict = self.generate_pose(pred_rotation, pred_centers, pred_depths, bbx_pred)

            return output_dict, segmentation
    

    def estimateTrans(self, translation_map, filter_bbx, pred_label):
        """
        translation_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        label: a tensor [batch_size, num_classes, height, width]
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            trans_map = translation_map[batch_id, (cls-1) * 3 : cls * 3, :]
            label = (pred_label[batch_id] == cls).detach()
            pred_T = trans_map[:, label].mean(dim=1)
            pred_Ts[idx] = pred_T
        return pred_Ts


    def gtTrans(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Ts[idx] = input_dict['RTs'][batch_id][cls - 1][:3, [3]].T
        return gt_Ts 


    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)


    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4 : cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)
        label = torch.tensor(label)
        return pred_Rs, label


    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs 


    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """        
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].numpy()
            center = pred_centers[bs, obj_id - 1].numpy()
            depth = pred_depths[bs, obj_id - 1].numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict


def eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_dict, label = model(inputdict)
    poselist = []
    rgb =  (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_dict[0],
        alpha = alpha
        )