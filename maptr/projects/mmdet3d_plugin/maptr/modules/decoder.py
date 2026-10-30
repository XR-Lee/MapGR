import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout=0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, bias=True, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        attention, _ = self.attention(query, key, value)
        out = self.mlp(attention + self.dropout(query))
        # query = self.norm1(attention + self.dropout(query))
        # out = self.mlp(query)
        # out = self.norm2(out + self.dropout(query))
        return out

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False


        ##################### TODO: reuse global embed  ########################
        self.global_embed = nn.Embedding(6*64, 512)

        self.cross_att = []
        self.self_att = []
        self.dim_up = []
        for i in range(6):
            self.cross_att.append(EncoderLayer(512,8,512))
            self.cross_att.append(EncoderLayer(512,8,512))
           
            self.self_att.append(EncoderLayer(512,8,512))
            self.self_att.append(EncoderLayer(512,8,512))
           
            self.dim_up.append(nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512)))
            
            self.dim_up.append(nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512)))

        self.cross_att = nn.ModuleList(self.cross_att)
        self.self_att = nn.ModuleList(self.self_att)
        self.dim_up = nn.ModuleList(self.dim_up)
        
        
        self.global_dec = []
        self.global_map = []
        
        self.global_dec_guide = []
        self.global_dec_guide_offest = []
        for i in range(6):            
            global_decoder = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 2048))
            
            global_guide = nn.Sequential(
                    nn.Linear(127*63, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512))
            
            global_guide_off = nn.Sequential(
                    nn.Linear(512 + 256, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256))
            
            global_map = nn.Sequential(
                    nn.ConvTranspose2d(50,32,kernel_size=3, stride=2,padding=1),
                    #nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    #nn.ConvTranspose2d(32,32,kernel_size=3, stride=2,padding=1),
                    nn.Conv2d(32,32,kernel_size=3,padding=1),
                    #nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Conv2d(32,1,kernel_size=3,padding=1))
            
            # global_map = nn.Sequential(
            #         nn.Conv2d(50,32,kernel_size=3, stride=2,padding=1),
            #         #nn.BatchNorm2d(32),
            #         nn.LeakyReLU(),
            #         nn.Conv2d(32,32,kernel_size=3, stride=2,padding=1),
            #         #nn.LayerNorm(1024),
            #         nn.LeakyReLU(),
            #         nn.Conv2d(32,1,kernel_size=3,padding=1))

            self.global_dec.append(global_decoder)
            self.global_map.append(global_map)
            self.global_dec_guide.append(global_guide)
            self.global_dec_guide_offest.append(global_guide_off)
        self.global_dec = nn.ModuleList(self.global_dec)
        self.global_map = nn.ModuleList(self.global_map)
        self.global_dec_guide = nn.ModuleList(self.global_dec_guide)
        self.global_dec_guide_offest = nn.ModuleList(self.global_dec_guide_offest)

        self.mask_emb = nn.Sequential(
                    nn.Linear(1, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512))
        self.pos_emb = nn.Embedding(200, 512)

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
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
        global_map_rec = []

        # rrmask = (torch.rand(4,1,4,2,device=output.device) > 0.5).float()
        # rrmask = F.interpolate(rrmask, (20,10), mode='nearest')
        # b = torch.nn.functional.unfold(rrmask, (1,1), stride=(1,1))
        # fea_emb = self.mask_emb(b.permute(0,2,1)) + self.pos_emb.weight.view(1,200,512).repeat(output.shape[1], 1, 1)
        # fea_emb = torch.mean(fea_emb.view(4, 40, 5, 512), dim=2)
        
        for lid, layer in enumerate(self.layers):  

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)

            #########################################################
            ii = lid
            bs = output.shape[1]
            vector_dim = output.shape[2]
            num_vec_one2one = 50
            num_pts_per_vec = 20
            num_vec = output.shape[0] // 20
            global_embed = self.global_embed.weight[ii*64:(ii+1)*64]
            
            tmp = output.permute(1, 0, 2)[:,:1000,:] #.view(bs,num_vec, num_pts_per_vec,-1)[:,:num_vec_one2one,:,:]
            tmp = tmp.detach() * 0.8 + tmp * 0.2
            # if ii == 0:
            #     global_embed = global_embed.view(1,64,-1).repeat(bs, 1, 1)
            # else:
            #     global_embed = qu

            #tmp_key = self.dim_up[ii*2](fea_emb) #self.dim_up[ii*2](tmp) 
            #tmp_value = self.dim_up[ii*2+1](fea_emb) #self.dim_up[ii*2+1](tmp)
            
            #qu = self.self_att[ii*2](global_embed, global_embed, global_embed)
            #qu = self.cross_att[ii*2](global_embed, tmp_key, tmp_value)
            
            #qu = self.self_att[ii*2+1](qu, qu, qu)
            #qu = self.cross_att[ii*2+1](qu, tmp_key, tmp_value)

            #qu = fea_emb
            qu = torch.mean(tmp.view(bs, 50, -1, 256), dim=2)
            # qu mlp -> mean -> mlp -> cat -> mlp
            global_map = self.global_dec[ii](qu)    
            global_map = self.global_map[ii](global_map.view(-1,50,64,32))
            global_map_rec.append(F.interpolate(global_map, (200,100), mode='bilinear'))
            
            ###################### guidance  before or after? first guided then decode?,  ##########################
            #global_emb = self.global_dec_guide[ii](qu)
            #global_emb = torch.mean(global_emb, dim=1).unsqueeze(0).repeat(output.shape[0],1,1)
            global_emb = self.global_dec_guide[ii](global_map.view(1, bs, -1)).repeat(output.shape[0],1,1)
            res = self.global_dec_guide_offest[ii](torch.cat([output, global_emb], dim=2))
            output = output + res
            #########################################################

            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                # assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), global_map_rec

        return output, reference_points, global_map_rec


# self.dim_up = []
# self.dim_up_cross_attn = []
# self.global_mha = []
# self.self_mha = []
# self.global_cross_attn = []
# self.dim_down_cross_attn = []
# self.layer_norm_list = []
# self.adp_fusion = []
# self.ffn = []
# for i in range(6):
#     self.global_mha.append(torch.nn.MultiheadAttention(512, 8, dropout=0.1, bias=True, batch_first=True))
#     self.global_mha.append(torch.nn.MultiheadAttention(512, 8, dropout=0.1, bias=True, batch_first=True))

#     self.self_mha.append(torch.nn.MultiheadAttention(512, 8, dropout=0.1, bias=True, batch_first=True))
#     #self.self_mha.append(torch.nn.MultiheadAttention(512, 8, dropout=0.1, bias=True, batch_first=True))

#     # self.global_cross_attn.append(torch.nn.MultiheadAttention(512, 8, dropout=0.1, bias=True, batch_first=True))
#     # self.layer_norm_list.append(nn.Sequential(nn.LayerNorm(512)))
#     # self.layer_norm_list.append(nn.Sequential(nn.LayerNorm(512)))
#     # self.layer_norm_list.append(nn.Sequential(nn.LayerNorm(512)))
#     # self.layer_norm_list.append(nn.Sequential(nn.LayerNorm(512)))
#     # self.layer_norm_list.append(nn.Sequential(nn.LayerNorm(512)))

#     self.ffn.append(nn.Sequential(
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512)))
#     self.ffn.append(nn.Sequential(
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512)))
#     self.ffn.append(nn.Sequential(
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512)))


#     self.dim_up.append(nn.Sequential(
#             nn.Linear(256, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512)))
    
#     self.dim_up.append(nn.Sequential(
#             nn.Linear(256, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 512)))

#     # self.dim_up_cross_attn.append(nn.Sequential(
#     #         nn.Linear(256, 512),
#     #         #nn.LayerNorm(512),
#     #         nn.LeakyReLU(),
#     #         nn.Linear(512, 512)))

#     # self.dim_down_cross_attn.append(nn.Sequential(
#     #         nn.Linear(512, 256),
#     #         #nn.LayerNorm(256),
#     #         nn.LeakyReLU(),
#     #         nn.Linear(256, 256)))


#     # self.adp_fusion.append(nn.Sequential(
#     #         nn.Linear(512, 128),
#     #         nn.LayerNorm(128),
#     #         nn.LeakyReLU(),
#     #         nn.Linear(128, 32),
#     #         nn.LeakyReLU(),
#     #         nn.Linear(32, 1))) 

# self.global_mha = nn.ModuleList(self.global_mha)
# self.self_mha = nn.ModuleList(self.self_mha)
# self.dim_up = nn.ModuleList(self.dim_up)
# # self.layer_norm_list = nn.ModuleList(self.layer_norm_list)
# self.ffn = nn.ModuleList(self.ffn)
# # self.dim_up_cross_attn = nn.ModuleList(self.dim_up_cross_attn)
# # self.global_cross_attn = nn.ModuleList(self.global_cross_attn)
# # self.dim_down_cross_attn = nn.ModuleList(self.dim_down_cross_attn)
# # self.adp_fusion = nn.ModuleList(self.adp_fusion)


#global_embed = self.layer_norm_list[ii*5](global_embed)
#################
#################
# qu_tmp, _ = self.global_mha[ii*2](global_embed, tmp_key, tmp_value)
# qu = global_embed + self.ffn[ii*3](qu_tmp)
# #qu = self.layer_norm_list[ii*5+1](qu)
# qu = self.ffn[ii*3+1](qu)

# qu_tmp, _ = self.self_mha[ii*1](qu, qu, qu)
# qu = qu + self.ffn[ii*3+2](qu_tmp)

# qu_tmp, _ = self.global_mha[ii*2+1](qu, tmp_key, tmp_value)
# qu = qu + qu_tmp
#################
#################
# qu = self.ffn[ii*3+1](qu)
# #qu = self.layer_norm_list[ii*5+2](qu + self.ffn[ii*3](qu_tmp))

# qu_tmp, _ = self.global_mha[ii*2+1](qu, tmp_key, tmp_value)
# qu = qu + qu_tmp
# #qu = self.layer_norm_list[ii*5+3](qu + self.ffn[ii*3+1](qu_tmp))
# qu = self.ffn[ii*3+2](qu)

# qu_tmp, _ = self.self_mha[ii*2+1](qu, qu, qu)
# qu = qu + qu_tmp
#qu = self.layer_norm_list[ii*5+4](qu + self.ffn[ii*3+2](qu_tmp))

# output_up = self.dim_up_cross_attn[ii](output.permute(1, 0, 2))
# res_fea, _ = self.global_cross_attn[ii](output_up, qu, qu)
# res_fea = res_fea - output_up
# offest = self.dim_down_cross_attn[ii](res_fea).permute(1, 0, 2)
# adp_weight = torch.sigmoid(self.adp_fusion[ii](res_fea).permute(1, 0, 2))
# output = output + offest * adp_weight * 0.

# bev_prior = kwargs['value'].permute(1,0,2).view(bs,200,100,256).permute(0,3,1,2)
# bev_prior = F.interpolate(bev_prior, (64, 32), mode='bilinear')
# global_map = self.global_map[ii](torch.cat([global_map.view(-1,64,64,32), bev_prior], dim=1))



@TRANSFORMER_LAYER.register_module()
class DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 num_vec=50,
                 num_pts_per_vec=20,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 8
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        # 
        num_vec = kwargs['num_vec']
        num_pts_per_vec = kwargs['num_pts_per_vec']
        for layer in self.operation_order:
            if layer == 'self_attn':
                # import ipdb;ipdb.set_trace()
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec,n_batch,n_dim).flatten(1,2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec,n_batch,n_dim).flatten(1,2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=kwargs['self_attn_mask'],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    # import ipdb;ipdb.set_trace()
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    attn_index += 1
                    identity = query
                else:
                    # import ipdb;ipdb.set_trace()
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec,n_batch,n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec,n_batch,n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    # import ipdb;ipdb.set_trace()
                    query = query.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    query_pos = query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    attn_index += 1
                    identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

