import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from timm.models.layers import trunc_normal_

import kornia


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class visiontransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super(visiontransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class Projector(nn.Module):
    def __init__(self, input_dim=384, projector_dim=384, out_dim=768, projector_kernel=3, projector_layers=2):
        super(Projector, self).__init__()
        self.projector_dim = projector_dim

        self.blocks = nn.Sequential(
            *[
                self.make_block(input_dim if i == 0 else projector_dim)
                for i in range(projector_layers)
            ]
        )
        self.proj = nn.Conv2d(projector_dim, out_dim, kernel_size=1)
        self.apply(self._init_weights)
        
    def make_block(self, in_channels):
        block = [
                nn.Conv2d(in_channels, self.projector_dim, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                TransposeLast(tranpose_dim=-3),
                nn.LayerNorm(self.projector_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            ]

        return nn.Sequential(*block)
        
    def add_residual(self, x, residual):
        if (
            residual is None
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        #elif isinstance(m, nn.LayerNorm):
            #nn.init.constant_(m.bias, 0)
            #nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)


    def forward(self, x):
        B, T, C = x.shape
        h_size, w_size = int(math.sqrt(T)), int(math.sqrt(T))
        x = x.transpose(1, 2).reshape(B, C, h_size, w_size)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual)
            residual = x
        x = self.proj(x)

        return x
    
    
class Predictor(nn.Module):
    def __init__(self, input_dim=768, predictor_dim=768, out_dim=768, predictor_layers=1):
        super(Predictor, self).__init__()
        self.predictor_dim = predictor_dim
        self.blocks = nn.Sequential(
            *[
                self.make_block(input_dim if i == 0 else predictor_dim)
                for i in range(predictor_layers)
            ]
        )

        self.proj = nn.Conv2d(predictor_dim, out_dim, kernel_size=1)
        self.apply(self._init_weights)
        
    def make_block(self, in_dim):
        block = [
                nn.Conv2d(in_dim, self.predictor_dim, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                TransposeLast(tranpose_dim=-3),
                nn.LayerNorm(self.predictor_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            ]

        return nn.Sequential(*block)
        
    def add_residual(self, x, residual):
        if (
            residual is None
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        #elif isinstance(m, nn.LayerNorm):
            #nn.init.constant_(m.bias, 0)
            #nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        #residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            #x = self.add_residual(x, residual)
            #residual = x
        x = self.proj(x)
        return x
    

class GTSA(nn.Module):
    """
    GTSA: Geometric Transformation Sensitive Arichtecture
    """
    def __init__(self, drop_path_rate=0., img_size=[224], 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        projector_dim=384, out_dim=768, predictor_dim=768, args=None):
        super(GTSA, self).__init__()
        self.args = args
        self.base_encoder = visiontransformer(drop_path_rate=0.1,
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, 
        depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, norm_layer=norm_layer)
        self.base_projector = Projector(input_dim=embed_dim, projector_dim=projector_dim, out_dim=out_dim)
        self.base_predictor = Predictor(input_dim=out_dim, predictor_dim=predictor_dim, out_dim=out_dim)

        self.momentum_encoder = visiontransformer(drop_path_rate=0.,
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, 
        depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, norm_layer=norm_layer)
        self.momentum_projector = Projector(input_dim=embed_dim, projector_dim=projector_dim, out_dim=out_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
            
        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
            
        self.embed_dim = self.base_encoder.embed_dim
        
        self.rot_predictor = nn.Sequential(nn.Linear(384, 768),
                                           nn.LayerNorm(768),
                                            nn.GELU(),  # first layer
                                            nn.Linear(768, 768),
                                            nn.LayerNorm(768),
                                            nn.GELU(),  # second layer
                                            nn.Linear(768, 128),
                                            nn.LayerNorm(128),
                                            nn.Linear(128, 4))  # output layer
        
    def base_MultiCropWrapper(self, x):
        
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        local_out_list, backbone_out_list = [], []
        for end_idx in idx_crops:
            _out = self.base_encoder(torch.cat(x[start_idx: end_idx]))
            backbone_out_list.append(_out)
            _out = self.base_projector(_out)
            _out = self.base_predictor(_out)
            local_out_list.append(_out)
            start_idx = end_idx

        return local_out_list, backbone_out_list
    
    def momentum_MultiCropWrapper(self, x):

        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        local_out_list = []
        for iteration_num, end_idx in enumerate(idx_crops):
            _out = self.momentum_encoder(torch.cat(x[start_idx: end_idx]))
            _out = self.momentum_projector(_out) 
            local_out_list.append(_out)
                
            start_idx = end_idx

        return local_out_list
    
    def roi_align(self, tea_out, stu_out, bbox):
        global_student_out_list = stu_out[0].chunk(self.args.num_crops[0])
        local_student_out_list = stu_out[1].chunk(self.args.num_crops[1])
        
        total_student_out_list = []
        for i in range(self.args.num_crops[0]):
            total_student_out_list.append(global_student_out_list[i])
        for i in range(self.args.num_crops[1]):
            total_student_out_list.append(local_student_out_list[i])
        
        total_teacher_out_list = tea_out[0].detach().chunk(self.args.num_crops[0])
        
        total_pc_loss, total_loss, count = 0, 0, 0
        for iq, q in enumerate(total_teacher_out_list):          
            for v in range(len(total_student_out_list)):  
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                    
                # roi
                B = q.shape[0]
                B_s, C_s, h_s, w_s = total_student_out_list[v].shape
                B_t, C_t, h_t, w_t = q.shape
                
                if (h_s*w_s) > 100:
                    output_size = self.args.roi_out_size[0]
                else:
                    output_size = self.args.roi_out_size[1]
                    
                tea_bbox_list = [bbox["gc"][i, iq, v, :].unsqueeze(dim=0) / self.args.patch_size for i in range(bbox["gc"].shape[0])]
                tea_aligned = roi_align(q, tea_bbox_list, output_size=output_size, aligned =True)

                stu_bbox_list = [bbox["all"][i, v, iq, :].unsqueeze(dim=0) / self.args.patch_size for i in range(bbox["all"].shape[0])]
                stu_aligned = roi_align(total_student_out_list[v], stu_bbox_list, output_size=output_size, aligned =True)
                
                offset = bbox["angles"][v] - bbox["angles"][iq]
                tea_aligned = kornia.geometry.transform.rotate(tea_aligned, offset)
                
                loss = self.cosine_loss(stu_aligned, tea_aligned)
                total_loss += loss.mean()
                count += 1
                
                #matching alogorithm
                stu_feats = total_student_out_list[v].reshape(B_s, C_s, h_s*w_s).permute(0, 2, 1)
                tea_feats = q.reshape(B_t, C_t, h_t*w_t).permute(0, 2, 1)
                
                if (h_s*w_s) > 100:
                    num_matches_on_l2 = int(self.args.num_matches[0])
                else:
                    num_matches_on_l2 = int(self.args.num_matches[1])

                maps_1_filtered, maps_1_nn = self.neirest_neighbores_on_l2(stu_feats, tea_feats, num_matches_on_l2)
                maps_1_filtered_sym, maps_1_nn_sym = self.neirest_neighbores_on_l2(tea_feats, stu_feats, num_matches_on_l2)
                
                pc_loss_1 = self.cosine_loss(maps_1_filtered, maps_1_nn)
                pc_loss_2 = self.cosine_loss(maps_1_filtered_sym, maps_1_nn_sym)
                total_pc_loss += 0.5 * (pc_loss_1.mean() + pc_loss_2.mean())
                 
        total_loss /= count 
        total_pc_loss /= count
       
        return total_loss, total_pc_loss
    
    def cosine_loss(self, feats_1, feats_2):
        if feats_1.dim() == 4:
            feats_1_normalized = F.normalize(feats_1, dim=1)
            feats_2_normalized = F.normalize(feats_2, dim=1)

            # Compute cosine similarity along the channel dimension (C) using einsum
            cos_similarity = torch.einsum('bchw,bchw->bhw', feats_1_normalized, feats_2_normalized)
            loss = - cos_similarity
            
        elif feats_1.dim() == 3:

            cosine_similarity = nn.CosineSimilarity(dim=-1)
            cos_similarity = cosine_similarity(feats_1, feats_2)
            loss = - cos_similarity

        else:
            print("Something wrong")
        
        return loss
            
    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def neirest_neighbores(self, input_maps, candidate_maps, distances, num_matches):
        batch_size = input_maps.size(0)

        if num_matches is None or num_matches == -1:
            num_matches = input_maps.size(1)

        topk_values, topk_indices = distances.topk(k=1, largest=False)
        topk_values = topk_values.squeeze(-1)
        topk_indices = topk_indices.squeeze(-1)

        sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
        sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

        mask = torch.stack(
            [
                torch.where(sorted_indices_indices[i] < num_matches, True, False)
                for i in range(batch_size)
            ]
        )
        topk_indices_selected = topk_indices.masked_select(mask)
        topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

        indices = (
            torch.arange(0, topk_values.size(1))
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(topk_values.device)
        )
        indices_selected = indices.masked_select(mask)
        indices_selected = indices_selected.reshape(batch_size, num_matches)

        filtered_input_maps = self.batched_index_select(input_maps, 1, indices_selected)
        filtered_candidate_maps = self.batched_index_select(
            candidate_maps, 1, topk_indices_selected
        )

        return filtered_input_maps, filtered_candidate_maps

    def neirest_neighbores_on_l2(self, input_maps, candidate_maps, num_matches):
        #distances = torch.cdist(input_maps, candidate_maps)

        # 각 Tensor를 정규화합니다.
        input_maps_normalized = F.normalize(input_maps, p=2, dim=-1)
        candidate_maps_normalized = F.normalize(candidate_maps, p=2, dim=-1)

        # Cosine similarity 계산
        cosine_similarities = -(torch.bmm(input_maps_normalized, candidate_maps_normalized.transpose(1, 2)))

        #return self.neirest_neighbores(input_maps, candidate_maps, distances, num_matches)
        return self.neirest_neighbores(input_maps, candidate_maps, cosine_similarities, num_matches)
    
    def rot_predict(self, stu_out, bbox):
        global_student_out_list = stu_out[0].chunk(self.args.num_crops[0])
        local_student_out_list = stu_out[1].chunk(self.args.num_crops[1])
        
        total_student_out_list = []
        for i in range(self.args.num_crops[0]):
            total_student_out_list.append(global_student_out_list[i])
        for i in range(self.args.num_crops[1]):
            total_student_out_list.append(local_student_out_list[i])
        
        total_loss = 0
        for v in range(len(total_student_out_list)):

            global_avg_output = total_student_out_list[v].mean(dim=1)
            rot_logits = self.rot_predictor(global_avg_output)
            
            pre_labels = bbox["angles"][v]
            #change pre_labels to rot_labels
            bbox["angles"][v][bbox["angles"][v] == 0] = 0
            bbox["angles"][v][bbox["angles"][v] == 90] = 1
            bbox["angles"][v][bbox["angles"][v] == 180] = 2
            bbox["angles"][v][bbox["angles"][v] == 270] = 3
            rot_labels = bbox["angles"][v].long()
            rot_loss = torch.nn.functional.cross_entropy(rot_logits, rot_labels)
            total_loss += rot_loss
        total_loss /= len(total_student_out_list)
        
        return total_loss

    def forward(self, x, bbox, m):
        stu_local_out_list, backbone_out_list = self.base_MultiCropWrapper(x)
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)# update the momentum encoder
            
        tea_local_out_list = self.momentum_MultiCropWrapper(x[:2])
        loss, pc_loss = self.roi_align(tea_local_out_list, stu_local_out_list, bbox)
        rot_pre_loss = self.rot_predict(backbone_out_list, bbox)
        
        return loss, pc_loss, rot_pre_loss
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
            
        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
            

def gtsa_small(img_size=[224], patch_size=16, drop_path_rate=0., projector_dim=384, predictor_dim=768, out_dim=768, args=None):
    model = GTSA(drop_path_rate=drop_path_rate,
        img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                projector_dim=projector_dim, predictor_dim=predictor_dim, out_dim=out_dim, args=args)
    return model

def gtsa_base(img_size=[224], patch_size=16, drop_path_rate=0., projector_dim=384, predictor_dim=768, out_dim=768, args=None):
    model = GTSA(drop_path_rate=drop_path_rate,
        img_size=img_size, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        projector_dim=projector_dim, predictor_dim=predictor_dim, out_dim=out_dim, args=args)
    return model
