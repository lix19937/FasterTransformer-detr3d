import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence, 
                                         build_positional_encoding)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER

# from mmcv.ops.point_sample import bilinear_grid_sample
from .utils import bilinear_grid_sample
import warnings

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class SVDetTransformer(BaseModule):
    """Implements the SVDet transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(SVDetTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the SVDetTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, SVDetCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `SVDetTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


    def forward_onnx(self,
                mlvl_feats,
                query_embed,
                img_shape,
                lidar2cam,
                #distortion,
                #intrinsic,
                pol_datas, cxy_cropxseyse_oxy,
                reg_branches=None,
                **kwargs):
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder.forward_onnx(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            img_shape=img_shape,
            lidar2cam=lidar2cam,
            #distortion = distortion,
            #intrinsic = intrinsic,
            pol_datas=pol_datas,
            cxy_cropxseyse_oxy=cxy_cropxseyse_oxy,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SVDetTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in SVDet transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(SVDetTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `SVDetTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

    def forward_onnx(self,
                     query,
                     *args,
                     img_shape=None,
                     lidar2cam=None,
                     # distortion=None,
                     # intrinsic=None,
                     pol_datas=None,
                     cxy_cropxseyse_oxy=None,
                     reference_points=None,
                     reg_branches=None,
                     **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer.forward(
                output,
                *args,
                img_shape=img_shape,
                lidar2cam=lidar2cam,
                # distortion=distortion,
                # intrinsic=intrinsic,
                pol_datas=pol_datas,
                cxy_cropxseyse_oxy=cxy_cropxseyse_oxy,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                                                ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                                                 ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points



@ATTENTION.register_module()
class SVDetCrossAtten(BaseModule):
    """An attention module used in SVDet. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 cam_embeds_cfg=None,
                 feat_embeds_cfg=None,
                 batch_first=False):
        super(SVDetCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.cam_embeds = self.feat_embeds = None
        if cam_embeds_cfg is not None:
            self.cam_embeds = build_positional_encoding(cam_embeds_cfg)
        if feat_embeds_cfg is not None:
            self.feat_embeds = build_positional_encoding(feat_embeds_cfg)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of SVDetCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        if 'img_metas' in kwargs:
            reference_points_3d, output, mask = feature_sampling(
                value, reference_points, self.pc_range, kwargs['img_metas'])
            output = torch.nan_to_num(output)
            mask = torch.nan_to_num(mask)
        elif 'img_shape' in kwargs and 'lidar2cam' in kwargs:
            reference_points_3d, output, mask = feature_sampling_onnx(
                value, reference_points, self.pc_range, kwargs['img_shape'], kwargs['lidar2cam'],
                kwargs['pol_datas'], kwargs['cxy_cropxseyse_oxy'])

        if self.cam_embeds:
            cam_embeds = self.cam_embeds(kwargs.get('img_metas', None))
            output = output + cam_embeds.permute(1, 0)[None, :, None, :, None, None]
        if self.feat_embeds:
            feat_embeds = self.feat_embeds()
            output = output + feat_embeds.permute(1, 0)[None, :, None, None, None, :]

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat





def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    distortion = []
    intrinsic = []

    pol_datas=[]
    center_xs=[]
    center_ys=[]
    crop_cfg_inputs = []
    dsize_inputs = []
    affine_Ms = []
    for img_meta in img_metas:
        #print('img_meta: ', img_meta.keys())
        #lidar2img.append(img_meta['lidar2img'])
        lidar2img.append(img_meta['lidar2cam'])
        distortion.append(img_meta['distortion'])
        #print('distortion: ', distortion)
        intrinsic.append(img_meta['cam_intrinsic'])

        pol_data=[]
        center_x=[]
        center_y=[]
        crop_cfg_input=[]
        dsize_input=[]
        for i in range(len(img_meta['cam_intrinsic_ocam'])):
          pol_data.append(img_meta['cam_intrinsic_ocam'][i]['pol_data'])
          center_x.append(img_meta['cam_intrinsic_ocam'][i]['center_x'])
          center_y.append(img_meta['cam_intrinsic_ocam'][i]['center_y'])
          crop_cfg_input.append(img_meta['crop_cfg_input'][i])
          dsize_input.append(img_meta['dsize_input'][i])
        pol_datas.append(pol_data)
        center_xs.append(center_x)
        center_ys.append(center_y)
        crop_cfg_inputs.append(crop_cfg_input)
        dsize_inputs.append(dsize_input)
        if ('affine_M' in img_meta.keys()):
          affine_Ms.append(img_meta['affine_M']) 

    lidar2img = np.asarray(lidar2img)
    distortion = np.asarray(distortion)
    intrinsic = np.asarray(intrinsic)

    pol_datas = np.asarray(pol_datas)
    center_xs = np.asarray(center_xs)
    center_ys = np.asarray(center_ys)

    crop_cfg_inputs = np.asarray(crop_cfg_inputs)
    dsize_inputs = np.asarray(dsize_inputs)

    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    distortion = reference_points.new_tensor(distortion)
    intrinsic = reference_points.new_tensor(intrinsic)

    pol_datas = reference_points.new_tensor(pol_datas) #(1,4,5)
    center_xs = reference_points.new_tensor(center_xs)
    center_ys = reference_points.new_tensor(center_ys)
    center_xy = torch.stack((center_xs, center_ys),-1)#(1,4,2)

    crop_cfg_inputs = reference_points.new_tensor(crop_cfg_inputs)
    dsize_inputs = reference_points.new_tensor(dsize_inputs)
    if ('affine_M' in img_meta.keys()):
      affine_Ms = torch.stack(affine_Ms).to(reference_points.device).to(reference_points.dtype)

    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    
    '''
    #equidistant model
    corners_flag = reference_points_cam[..., 2:3] > 0

    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    r = torch.norm(reference_points_cam[..., :2], dim=-1, keepdim=True)
    theta = torch.arctan(r)
    theta = torch.where(corners_flag, theta, 3.14 - theta)

    theta_d = theta + theta**3 * distortion[:, :, 0:1, None] + theta**5 * distortion[:, :, 1:2, None] + theta**7 * distortion[:, :, 2:3, None] + theta**9 * distortion[:, :, 3:4, None]
    inv_r = torch.where(r > eps, 1./r, torch.ones_like(r))
    cdist = torch.where(r > eps, theta_d * inv_r, torch.ones_like(r))
    reference_points_cam = reference_points_cam * cdist
    reference_points_cam = torch.cat((reference_points_cam, torch.ones_like(reference_points_cam)), -1).view(B, num_cam, num_query, 4, 1)
    intrinsic = intrinsic.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(intrinsic, reference_points_cam).squeeze(-1)[..., :2]
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    #equidistant model
    '''
    #ocam model
    corners_flag = reference_points_cam[..., 2:3] > 0
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    r = torch.norm(reference_points_cam[..., :2], dim=-1, keepdim=True)
    theta = torch.arctan(r)
    theta = torch.where(corners_flag, theta, 3.14 - theta)

    r_d = theta*pol_datas[:, :, 0:1, None] + theta**2 * pol_datas[:, :, 1:2, None] + theta**3 * pol_datas[:, :, 2:3, None] + theta**4 * pol_datas[:, :, 3:4, None] + theta**5 * pol_datas[:, :, 4:5, None]
    inv_r = torch.where(r > eps, 1./r, torch.ones_like(r))
    cdist = torch.where(r > eps, r_d * inv_r, torch.ones_like(r))
    reference_points_cam = reference_points_cam * cdist
    #print('reference_points_cam: ',reference_points_cam.shape, center_xy.shape)  #[1, 4, 10, 2]
    #[1, 4, 10, 2] [1, 4, 2]
    #reference_points_cam[..., 0]=img_metas[0]['img_shape'][0][1]-(reference_points_cam[..., 0]+center_xy[...,0:1])
    #reference_points_cam[..., 1] = img_metas[0]['img_shape'][0][0] - (reference_points_cam[..., 1] + center_xy[..., 1:2])
    #reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    #reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    #reference_points_cam[..., 0] = center_xy[..., 0:1]*2 - (reference_points_cam[..., 0] + center_xy[...,0:1])
    #reference_points_cam[..., 1] = center_xy[..., 1:2]*2 - (reference_points_cam[..., 1] + center_xy[..., 1:2])
    #reference_points_cam[..., 0] /= center_xy[..., 0:1]*2 
    #reference_points_cam[..., 1] /= center_xy[..., 1:2]*2
    reference_points_cam[..., 0] = dsize_inputs[...,0:1] - (reference_points_cam[..., 0]+center_xy[...,0:1])
    reference_points_cam[..., 1] = dsize_inputs[...,1:2] - (reference_points_cam[..., 1] + center_xy[..., 1:2])

    crop_x_start = crop_cfg_inputs[:, :, 1, 0] * dsize_inputs[:, :, 0]
    crop_x_end = crop_cfg_inputs[:, :, 1, 1] * dsize_inputs[:, :, 0]
    crop_y_start = crop_cfg_inputs[:, :, 0, 0] * dsize_inputs[:, :, 1]
    crop_y_end = crop_cfg_inputs[:, :, 0, 1] * dsize_inputs[:, :, 1]  #(1,4,1)
    #print('shape: ',reference_points_cam.shape, crop_x_start.shape, crop_cfg_inputs.shape, dsize_inputs.shape)
    #[1, 4, 10, 2] [1, 4, 4, 1] [1, 4, 2, 2] [1, 4, 2]
    reference_points_cam[..., 0] = reference_points_cam[..., 0] - crop_x_start.unsqueeze(-1)
    reference_points_cam[..., 1] = reference_points_cam[..., 1] - crop_y_start.unsqueeze(-1)
    scale_x = img_metas[0]['img_shape'][0][1] / (crop_x_end - crop_x_start)
    scale_y = img_metas[0]['img_shape'][0][0] / (crop_y_end - crop_y_start)
    #print('scale_xy: ',scale_x.shape, scale_y.shape)  #[1,4]
    reference_points_cam[..., 0] = reference_points_cam[..., 0] * scale_x.unsqueeze(-1)
    reference_points_cam[..., 1] = reference_points_cam[..., 1] * scale_y.unsqueeze(-1)

    #----------ImageAug3D----------
    if ('affine_M' in img_meta.keys()):
      # affine_Ms:[1, 4, 2, 3]  reference_points_cam:[1, 4, 10, 2]
      reference_points_cam = torch.cat( (reference_points_cam, torch.ones_like(reference_points_cam[...,:1])), -1)
      reference_points_cam = reference_points_cam.transpose(3 , 2)
      #print('affine_Ms, reference_points_cam: ', affine_Ms.shape, reference_points_cam.shape)
      reference_points_cam = torch.matmul(affine_Ms, reference_points_cam)
      reference_points_cam = reference_points_cam.transpose(3,2)

    #print('reference_points_cam: ', reference_points_cam)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    #ocam model

    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    
    mask = mask & corners_flag
    
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    for bs_id, img_meta in enumerate(img_metas):
        if 'valid_cameras' in img_meta:
            mask[bs_id, :, :, ~torch.tensor(img_meta['valid_cameras']), ...] = 0
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # sampled_feat = bilinear_grid_sample(feat, reference_points_cam_lvl) # slower than F.grid_sample
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask



def feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2cam, pol_datas, cxy_cropxseyse_oxy):
    #print('img_shape: ', img_shape)  #[[288, 736]]
    lidar2cam = lidar2cam.type_as(mlvl_feats[0])


    pol_datas = pol_datas.type_as(mlvl_feats[0])
    cxy_cropxseyse_oxy = cxy_cropxseyse_oxy.type_as(mlvl_feats[0])

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
    num_cam = lidar2cam.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2cam = lidar2cam.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    reference_points_cam = torch.matmul(lidar2cam, reference_points).squeeze(-1)
    corners_flag = reference_points_cam[..., 2:3] > 0
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    #reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #    reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.clamp(reference_points_cam[..., 2:3], min=0.01)
    r = torch.norm(reference_points_cam[..., :2], dim=-1, keepdim=True)
    theta = torch.arctan(r)
    theta = torch.where(corners_flag, theta, 3.14 - theta)

    theta_d = theta*pol_datas[:, :, 0:1, None] + theta**2 * pol_datas[:, :, 1:2, None] + theta**3 * pol_datas[:, :, 2:3, None] + \
              theta**4 * pol_datas[:, :, 3:4, None] + theta**5 * pol_datas[:, :, 4:5, None]

    inv_r = torch.where(r > eps, 1./r, torch.ones_like(r))
    cdist = torch.where(r > eps, theta_d * inv_r, torch.ones_like(r))
    reference_points_cam = reference_points_cam * cdist

    reference_points_cam[..., 0] = ori_inputs[...,0:1] - (reference_points_cam[..., 0] + center_xy[..., 0:1])
    reference_points_cam[..., 1] = ori_inputs[...,1:2] - (reference_points_cam[..., 1] + center_xy[..., 1:2])


    crop_x_start = crop_cfg_inputs[:, :, 0] * ori_inputs[:, :, 0]
    crop_x_end = crop_cfg_inputs[:, :, 1] * ori_inputs[:, :, 0]
    crop_y_start = crop_cfg_inputs[:, :, 2] * ori_inputs[:, :, 1]
    crop_y_end = crop_cfg_inputs[:, :, 3] * ori_inputs[:, :, 1]  #[1, 4]

    scale_x = img_shape[0][1] / (crop_x_end - crop_x_start)
    scale_y = img_shape[0][0] / (crop_y_end - crop_y_start)

    reference_points_cam[..., 0] = reference_points_cam[..., 0] - crop_x_start.unsqueeze(-1)
    reference_points_cam[..., 1] = reference_points_cam[..., 1] - crop_y_start.unsqueeze(-1)

    reference_points_cam[..., 0] = reference_points_cam[..., 0] * scale_x.unsqueeze(-1)
    reference_points_cam[..., 1] = reference_points_cam[..., 1] * scale_y.unsqueeze(-1)

    #img_shape:  tensor([[288, 736]])
    reference_points_cam[..., 0] /= img_shape[0][1]
    reference_points_cam[..., 1] /= img_shape[0][0]

    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask & corners_flag
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)

    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        # sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = bilinear_grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask