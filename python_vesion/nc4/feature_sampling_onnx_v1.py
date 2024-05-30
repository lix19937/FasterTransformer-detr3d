
import torch
from  utils import bilinear_grid_sample, savetensor_byrow, loadtxt

# base feature_sampling_onnx.py

# pc_range :  [-20.2, -20.2, -5.0, 20.2, 20.2, 3.0] 

def feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img, pol_datas, cxy_cropxseyse_oxy):
    #print('img_shape: ', img_shape)  #[[288, 736]]
    lidar2img = lidar2img.type_as(mlvl_feats[0])# [4,4,4]
    pol_datas = pol_datas.type_as(mlvl_feats[0])#[4,5]
    cxy_cropxseyse_oxy = cxy_cropxseyse_oxy.type_as(mlvl_feats[0])#[4,8]

    center_xy = cxy_cropxseyse_oxy[...,:2]
    crop_cfg_inputs = cxy_cropxseyse_oxy[...,2:6]  #crop_x_start crop_x_end crop_y_start crop_y_end
    ori_inputs = cxy_cropxseyse_oxy[...,6:]

    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)                           #[4, 4, 4]
    print("reference_points shape",reference_points.shape)#[1, 512, 4]
    savetensor_byrow(reference_points.permute(0,2, 1), 'reference_points_permute-aa.data', fmt = "%.6f")

######################### begin
    reference_points_z = reference_points.view(-1, 4).permute(1, 0)# [4, 512]
    lidar2img_z = lidar2img.view(-1, 4)# [16, 4]
    reference_points_cam_x = torch.matmul(lidar2img_z, reference_points_z)# [16, 512] -> [1, 4, 4, 512]  means [1, NC, 4, L] 
    reference_points_cam_z = reference_points_cam_x.view(1, num_cam, 4, num_query).permute(0, 1, 3, 2) #  [1, NC, L, 4] 
    corners_flag_z = reference_points_cam_x.view(1, num_cam, 4, num_query)[..., 2:3, :] > 0
    print("corners_flag_z shape", corners_flag_z.shape) #[1, 4, 1, 512]
######################### end 

    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    eps = 1e-5
    print("lidar2img shape",lidar2img.shape)#[1, 4, 512, 4, 4]
    print("reference_points shape",reference_points.shape)#[1, 4, 512, 4, 1]

    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    print("reference_points_cam shape",reference_points_cam.shape)#[1, 4, 512, 4]
    print(">>>>", torch.equal(reference_points_cam_z, reference_points_cam))
    savetensor_byrow(reference_points_cam.permute(0,1,3,2), 'reference_points_cam_permute-bb.data', fmt = "%.6f")

    # savetensor_byrow(reference_points_cam, 'reference_points_cam.data', fmt = "%.6f")
    # savetensor_byrow(reference_points_cam_z, 'reference_points_cam_z.data', fmt = "%.6f")
##########
    corners_flag = reference_points_cam[..., 2:3] > 0
    mask = (reference_points_cam[..., 2:3] > 1e-5)
    print("corners_flag shape", corners_flag.shape) #[1, 4, 512, 1]
    print(">>>>", torch.equal(corners_flag, corners_flag_z.view(corners_flag.shape)))
    # savetensor_byrow(corners_flag, 'corners_flag.data', fmt = "%.6f")
    # savetensor_byrow(corners_flag_z.view(corners_flag.shape), 'corners_flag_z.data', fmt = "%.6f")

    reference_points_cam = reference_points_cam[..., 0:2] / torch.clamp(reference_points_cam[..., 2:3], min=0.01)
######################################
    print("reference_points_cam shape", reference_points_cam.shape) #  [1, 4, 512, 2]
    r = torch.norm(reference_points_cam[..., :2], dim=-1, keepdim=True) # based on row 
    print("r shape", r.shape) # [1, 4, 512, 1]

    theta = torch.arctan(r)
    theta = torch.where(corners_flag, theta, 3.14 - theta)
    # pol_datas shape here expland  [1,4,5,1]  -> [1,4,1,1]
    theta_d = theta*     pol_datas[:, :, 0:1, None] + \
              theta**2 * pol_datas[:, :, 1:2, None] + \
              theta**3 * pol_datas[:, :, 2:3, None] + \
              theta**4 * pol_datas[:, :, 3:4, None] + \
              theta**5 * pol_datas[:, :, 4:5, None]

    inv_r = torch.where(r > eps, 1./r,            torch.ones_like(r))
    cdist = torch.where(r > eps, theta_d * inv_r, torch.ones_like(r))
    print("--reference_points_cam shape", reference_points_cam.shape) # [1, 4, 512, 2]
    print("--cdist shape", cdist.shape) # [1, 4, 512, 1]

    reference_points_cam = reference_points_cam * cdist
    print("--reference_points_cam shape", reference_points_cam.shape) # [1, 4, 512, 2]

    reference_points_cam[..., 0] = ori_inputs[...,0:1] - (reference_points_cam[..., 0] + center_xy[..., 0:1])
    reference_points_cam[..., 1] = ori_inputs[...,1:2] - (reference_points_cam[..., 1] + center_xy[..., 1:2])
    print("--reference_points_cam shape", reference_points_cam.shape)  # [1, 4, 2, 512]


    crop_x_start = crop_cfg_inputs[:, :, 0] * ori_inputs[:, :, 0]
    crop_x_end   = crop_cfg_inputs[:, :, 1] * ori_inputs[:, :, 0]
    crop_y_start = crop_cfg_inputs[:, :, 2] * ori_inputs[:, :, 1]
    crop_y_end   = crop_cfg_inputs[:, :, 3] * ori_inputs[:, :, 1]  #[1, 4]

    scale_x = img_shape[0][1] / (crop_x_end - crop_x_start)
    scale_y = img_shape[0][0] / (crop_y_end - crop_y_start)
    print("scale_x.shape", scale_x.unsqueeze(-1).shape) # [1, 4, 1]
    print("scale_y.shape", scale_y.unsqueeze(-1).shape) # [1, 4, 1]
    print("reference_points_cam[..., 0].shape", reference_points_cam[..., 0].shape) # [1, 4, 512]

    reference_points_cam[..., 0] = (reference_points_cam[..., 0] - crop_x_start.unsqueeze(-1)) * scale_x.unsqueeze(-1) /img_shape[0][1]
    reference_points_cam[..., 1] = (reference_points_cam[..., 1] - crop_y_start.unsqueeze(-1)) * scale_y.unsqueeze(-1) /img_shape[0][0]
    # savetensor_byrow(reference_points_cam, 'reference_points_cam_v1.data', fmt = "%.6f")
    # exit(0)
    #img_shape:  tensor([[288, 736]])
#########################################

    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask & corners_flag
    print("mask.shape", mask.shape) # [1, 4, 512, 1]
    savetensor_byrow(reference_points_cam, 'reference_points_cam_v1.data', fmt = "%.6f")

    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    print("mask.shape", mask.shape) # [1, 1, 512, 4, 1, 1]
    savetensor_byrow(mask, 'mask_v1.data', fmt = "%d")
    # exit(0)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = bilinear_grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask  


def main():

  bp = "/home/igs/avod/0413/fs/"
  cxy_cropxseyse_oxy = loadtxt(file_name=bp+"ca.fs.In.cxy_cropxseyse_oxy.1-4-8", shape=(1,4,8))
  img_shape = loadtxt(file_name=bp+"ca.fs.In.img_shape.1-2", shape=(1,2))
  print(img_shape)

  lidar2img = loadtxt(file_name=bp+"ca.fs.In.lidar2cam.1-4-4-4", shape=(1,4,4,4))
  mlvl_feats = []
  a = loadtxt(file_name=bp+"ca.fs.In.mlvl_feats.0.1-4-256-72-184", shape=(1,4,256,72,184))
  b = loadtxt(file_name=bp+"ca.fs.In.mlvl_feats.1.1-4-256-36-92", shape=(1,4,256,36,92))
  c = loadtxt(file_name=bp+"ca.fs.In.mlvl_feats.2.1-4-256-18-46", shape=(1,4,256,18,46))
  d = loadtxt(file_name=bp+"ca.fs.In.mlvl_feats.3.1-4-256-9-23", shape=(1,4,256,9,23))
  mlvl_feats.append(a)
  mlvl_feats.append(b)
  mlvl_feats.append(c)
  mlvl_feats.append(d)

  reference_points = loadtxt(file_name=bp+"ca.fs.In.reference_points.1-512-3", shape=(1,512,3))
  pc_range = loadtxt(file_name=bp+"ca.fs.In.pc_range.1-6", shape=(1,6)).squeeze()
  print(pc_range.shape)

  pol_datas = loadtxt(file_name=bp+"ca.fs.In.pol_datas.1-4-5", shape=(1,4,5))

  reference_points_3d, sampled_feats, mask = feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img, pol_datas, cxy_cropxseyse_oxy)
  savetensor_byrow(reference_points_3d, "reference_points_3d", fmt = "%.6f")
  savetensor_byrow(sampled_feats, "sampled_feats", fmt = "%.6f")
  savetensor_byrow(mask, "mask", fmt = "%.6f")

  print("reference_points_3d shape", reference_points_3d.shape)
  print("sampled_feats shape", sampled_feats.shape)
  print("mask shape", mask.shape)

  print("=========done========")

if __name__ == '__main__':
  main()