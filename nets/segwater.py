#完整的
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

try:
    from mmcv.ops import DeformConv2d
    MMCV_AVAILABLE = True
except ImportError:
    MMCV_AVAILABLE = False
    print("Warning: Failed to import DeformConv2d from mmcv.ops. Ensure mmcv-full is installed correctly.")
    print("DCN in ASPP will fallback to standard dilated convolutions if mmcv is not available.")
    DeformConv2d = None

# -------------------------------------------------------------
# SELayer (Squeeze-and-Excitation Layer)
# -------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -------------------------------------------------------------
# Connect Module
# -------------------------------------------------------------
class Connect(nn.Module):
    def __init__(self, num_classes, num_neighbor, embedding_dim=768, dropout_ratio=0.1):
        super(Connect, self).__init__()
        self.seg_branch = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        )
        self.connect_branch = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_neighbor, 3, padding=1, dilation=1),
        )
        self.se = SELayer(num_neighbor)
        self.connect_branch_d1 = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_neighbor, 3, padding=3, dilation=3),
        )
        self.se_d1 = SELayer(num_neighbor)
        self._init_weight()

    def forward(self, input):
        seg = self.seg_branch(input)
        con = self.connect_branch(input)
        con0 = self.se(con)
        con_d1 = self.connect_branch_d1(input)
        con1 = self.se_d1(con_d1)
        return seg, con0, con1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# -------------------------------------------------------------
# MLP for Linear Embedding
# -------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x_shape = x.shape
        x_permuted = x.flatten(2).transpose(1, 2)
        x_projected = self.proj(x_permuted)
        return x_projected.transpose(1, 2).reshape(x_shape[0], self.proj.out_features, x_shape[2], x_shape[3]).contiguous()

# -------------------------------------------------------------
# DCP module
# -------------------------------------------------------------
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_dcn=True):
        super(ASPPModule, self).__init__()
        self.use_dcn_effective = use_dcn and MMCV_AVAILABLE
        rates = [1, 6, 12, 18]
        self.conv_list = nn.ModuleList()
        self.conv_list.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        for rate in rates:
            if self.use_dcn_effective and rate > 1:
                offset_channels_dcn = 1 * 2 * 3 * 3
                offset_generator = nn.Conv2d(in_channels, offset_channels_dcn,
                                             kernel_size=3, padding=rate, dilation=rate, bias=True)
                nn.init.constant_(offset_generator.weight, 0.)
                nn.init.constant_(offset_generator.bias, 0.)
                dcn_layer = DeformConv2d(in_channels, out_channels, kernel_size=3,
                                         padding=rate, dilation=rate, bias=False, deform_groups=1)

                class DeformableBranch(nn.Module):
                    def __init__(self, offset_gen, dcn_op):
                        super().__init__()
                        self.offset_gen = offset_gen
                        self.dcn_op = dcn_op

                    def forward(self, x_in):
                        offset = self.offset_gen(x_in)
                        return self.dcn_op(x_in, offset)

                self.conv_list.append(
                    nn.Sequential(
                        DeformableBranch(offset_generator, dcn_layer),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        num_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        input_h, input_w = x.shape[-2:]
        for conv_module_seq in self.conv_list:
            res.append(conv_module_seq(x))
        pooled_features = self.image_pool(x)
        pooled_features_upsampled = F.interpolate(pooled_features, size=(input_h, input_w), mode='bilinear', align_corners=False)
        res.append(pooled_features_upsampled)
        concatenated_features = torch.cat(res, dim=1)
        return self.project(concatenated_features)

# -------------------------------------------------------------
# SPCII Module Definitions
# -------------------------------------------------------------
class ChannelInteraction(nn.Module):
    def __init__(self, k_size=5):
        super(ChannelInteraction, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)

    def forward(self, x_channel_sequence):
        y = x_channel_sequence.unsqueeze(1)
        y = self.conv1d(y)
        return y.squeeze(1)

class SPCII_SpaceEmbeddedChannelModule(nn.Module):
    def __init__(self, channels):
        super(SPCII_SpaceEmbeddedChannelModule, self).__init__()
        self.channels = channels
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv_cat_avg = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_cat_max = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def _reshape_and_pool_if_needed(self, tensor_to_match_dim, target_dim_size, is_height_target):
        b, c, current_spatial_dim = tensor_to_match_dim.shape
        if current_spatial_dim == target_dim_size:
            return tensor_to_match_dim
        if is_height_target:
            return F.adaptive_avg_pool2d(tensor_to_match_dim.unsqueeze(-1), (target_dim_size, 1)).squeeze(-1)
        else:
            return F.adaptive_avg_pool2d(tensor_to_match_dim.unsqueeze(-1).permute(0, 1, 3, 2), (1, target_dim_size)).squeeze(-2)

    def forward(self, x):
        b, c, h, w = x.shape
        F_avg_H_orig = self.avg_pool_h(x).squeeze(-1)
        F_max_H_orig = self.max_pool_h(x).squeeze(-1)
        F_avg_W_orig = self.avg_pool_w(x).squeeze(-2)
        F_max_W_orig = self.max_pool_w(x).squeeze(-2)

        concat_avg = torch.cat((F_avg_W_orig, F_avg_H_orig), dim=2)
        F_HW_prime_avg = self.conv_cat_avg(concat_avg)
        F_W_prime_avg_path, F_H_prime_avg_path = torch.split(F_HW_prime_avg, [w, h], dim=2)

        concat_max = torch.cat((F_max_W_orig, F_max_H_orig), dim=2)
        F_HW_prime_max = self.conv_cat_max(concat_max)
        F_W_prime_max_path, F_H_prime_max_path = torch.split(F_HW_prime_max, [w, h], dim=2)

        term1_fh = F_H_prime_avg_path
        term2_fh_raw_w = F_W_prime_max_path
        term2_fh_processed = self._reshape_and_pool_if_needed(term2_fh_raw_w, h, is_height_target=True)
        f_h = (term1_fh + term2_fh_processed).unsqueeze(-1)

        term1_fw = F_W_prime_avg_path
        term2_fw_raw_h = F_H_prime_max_path
        term2_fw_processed = self._reshape_and_pool_if_needed(term2_fw_raw_h, w, is_height_target=False)
        f_w = (term1_fw + term2_fw_processed).unsqueeze(-2)
        return f_h, f_w

class SPCII_ChannelEmbeddedSpaceModule(nn.Module):
    def __init__(self, channels, b_param=1, gamma_param=2, num_groups_1d_conv=1):
        super(SPCII_ChannelEmbeddedSpaceModule, self).__init__()
        self.channels = channels
        self.num_groups_1d_conv = num_groups_1d_conv
        k_float = (math.log2(channels) + b_param) / gamma_param
        k = int(k_float + 0.5)
        if k % 2 == 0:
            k += 1
        self.k_size = k
        if self.num_groups_1d_conv > 1 and channels % self.num_groups_1d_conv == 0:
            self.group_channels = channels // self.num_groups_1d_conv
            self.interactions_h = nn.ModuleList(
                [ChannelInteraction(k_size=self.k_size) for _ in range(self.num_groups_1d_conv)]
            )
            self.interactions_w = nn.ModuleList(
                [ChannelInteraction(k_size=self.k_size) for _ in range(self.num_groups_1d_conv)]
            )
        elif self.num_groups_1d_conv == 1:
            self.interaction_h_single = ChannelInteraction(k_size=self.k_size)
            self.interaction_w_single = ChannelInteraction(k_size=self.k_size)
        else:
            if self.num_groups_1d_conv > 1:
                print(f"Warning: channels ({channels}) not divisible by num_groups_1d_conv ({self.num_groups_1d_conv}). Using num_groups_1d_conv=1.")
            self.num_groups_1d_conv = 1
            self.interaction_h_single = ChannelInteraction(k_size=self.k_size)
            self.interaction_w_single = ChannelInteraction(k_size=self.k_size)
        self.sigmoid = nn.Sigmoid()

    def _apply_channel_interaction(self, f_transformed, interactions_list_or_single, b_dim, spatial_dim_size):
        c_dim_input = f_transformed.shape[-1]
        if self.num_groups_1d_conv > 1 and c_dim_input % self.num_groups_1d_conv == 0:
            group_channels = c_dim_input // self.num_groups_1d_conv
            f_grouped = f_transformed.view(b_dim * spatial_dim_size, self.num_groups_1d_conv, group_channels)
            out_groups = []
            for i in range(self.num_groups_1d_conv):
                group_data = f_grouped[:, i, :]
                out_groups.append(interactions_list_or_single[i](group_data))
            raw_attention = torch.cat(out_groups, dim=1)
        else:
            raw_attention = interactions_list_or_single(f_transformed)
        return raw_attention

    def forward(self, f_h, f_w):
        b, c, h, _ = f_h.shape
        _, _, _, w = f_w.shape
        f_h_transformed = f_h.squeeze(-1).permute(0, 2, 1).contiguous().view(b * h, c)
        f_w_transformed = f_w.squeeze(-2).permute(0, 2, 1).contiguous().view(b * w, c)
        raw_attention_h = self._apply_channel_interaction(
            f_h_transformed,
            self.interactions_h if self.num_groups_1d_conv > 1 and c % self.num_groups_1d_conv == 0 else self.interaction_h_single,
            b, h
        )
        raw_attention_w = self._apply_channel_interaction(
            f_w_transformed,
            self.interactions_w if self.num_groups_1d_conv > 1 and c % self.num_groups_1d_conv == 0 else self.interaction_w_single,
            b, w
        )
        g_h = self.sigmoid(raw_attention_h).view(b, h, c).permute(0, 2, 1).unsqueeze(-1)
        g_w = self.sigmoid(raw_attention_w).view(b, w, c).permute(0, 2, 1).unsqueeze(-2)
        return g_h, g_w

class SPCII_Attention(nn.Module):
    def __init__(self, channels, b_eca=1, gamma_eca=2, num_groups_1d_conv=1):
        super(SPCII_Attention, self).__init__()
        self.part1_space_embedded_channel = SPCII_SpaceEmbeddedChannelModule(channels)
        self.part2_channel_embedded_space = SPCII_ChannelEmbeddedSpaceModule(channels,
                                                                             b_param=b_eca,
                                                                             gamma_param=gamma_eca,
                                                                             num_groups_1d_conv=num_groups_1d_conv)

    def forward(self, x):
        f_h, f_w = self.part1_space_embedded_channel(x)
        g_h, g_w = self.part2_channel_embedded_space(f_h, f_w)
        return x * g_h * g_w

# -------------------------------------------------------------
# DASCIModule: Deformable ASPP and Spatial-Channel Interaction Module
# -------------------------------------------------------------
class DASCIModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_dcn=True, num_groups_1d_conv_in_spcii=1):
        super(DASCIModule, self).__init__()
        self.aspp_module = ASPPModule(in_channels, out_channels, use_dcn=use_dcn)
        self.attention_module = SPCII_Attention(out_channels, num_groups_1d_conv=num_groups_1d_conv_in_spcii)

    def forward(self, x):
        x = self.aspp_module(x)
        x = self.attention_module(x)
        return x

# -------------------------------------------------------------
# SegformerHead (Modified with DASCIModule)
# -------------------------------------------------------------
class SegformerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1,
                 use_dcn_in_aspp=True, use_dasci_post_fusion=True,
                 dasci_individual_mlp_stages={'c1': False, 'c2': False, 'c3': False, 'c4': False},
                 num_groups_1d_conv_in_spcii=1):
        super(SegformerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.use_dasci_post_fusion = use_dasci_post_fusion
        self.dasci_individual_mlp_stages = dasci_individual_mlp_stages

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dasci_c4 = DASCIModule(embedding_dim, embedding_dim, use_dcn=use_dcn_in_aspp, num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii) \
            if self.dasci_individual_mlp_stages.get('c4', False) else nn.Identity()
        self.dasci_c3 = DASCIModule(embedding_dim, embedding_dim, use_dcn=use_dcn_in_aspp, num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii) \
            if self.dasci_individual_mlp_stages.get('c3', False) else nn.Identity()
        self.dasci_c2 = DASCIModule(embedding_dim, embedding_dim, use_dcn=use_dcn_in_aspp, num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii) \
            if self.dasci_individual_mlp_stages.get('c2', False) else nn.Identity()
        self.dasci_c1 = DASCIModule(embedding_dim, embedding_dim, use_dcn=use_dcn_in_aspp, num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii) \
            if self.dasci_individual_mlp_stages.get('c1', False) else nn.Identity()

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.post_fuse_dasci_module = DASCIModule(embedding_dim, embedding_dim, use_dcn=use_dcn_in_aspp, num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii) \
            if self.use_dasci_post_fusion else nn.Identity()

        self.con = Connect(num_classes, num_neighbor=9, embedding_dim=embedding_dim, dropout_ratio=dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h_c1, w_c1 = c1.size()

        _c4_mlp = self.linear_c4(c4)
        _c4_mlp = self.dasci_c4(_c4_mlp)
        _c4_up = F.interpolate(_c4_mlp, size=(h_c1, w_c1), mode='bilinear', align_corners=False)

        _c3_mlp = self.linear_c3(c3)
        _c3_mlp = self.dasci_c3(_c3_mlp)
        _c3_up = F.interpolate(_c3_mlp, size=(h_c1, w_c1), mode='bilinear', align_corners=False)

        _c2_mlp = self.linear_c2(c2)
        _c2_mlp = self.dasci_c2(_c2_mlp)
        _c2_up = F.interpolate(_c2_mlp, size=(h_c1, w_c1), mode='bilinear', align_corners=False)

        _c1_mlp = self.linear_c1(c1)
        _c1_mlp = self.dasci_c1(_c1_mlp)

        fused_features = self.linear_fuse(torch.cat([_c1_mlp, _c2_up, _c3_up, _c4_up], dim=1))
        fused_after_dasci = self.post_fuse_dasci_module(fused_features)

        seg_pred, con0_pred, con1_pred = self.con(fused_after_dasci)
        return seg_pred, con0_pred, con1_pred

# -------------------------------------------------------------
# SegFormer (Updated with DASCI Parameters)
# -------------------------------------------------------------
class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b2', pretrained=False, mit_weights_path=None,
                 use_gradient_checkpointing=False, use_dcn_in_aspp=True, use_dasci_post_fusion=True,
                 dasci_individual_mlp_stages={'c1': False, 'c2': False, 'c3': False, 'c4': False},
                 num_groups_1d_conv_in_spcii=1):
        super(SegFormer, self).__init__()
        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if phi == 'b0':
            self.encoder = mit_b0(pretrained=pretrained)
            encoder_channels = [32, 64, 160, 256]
        elif phi == 'b1':
            self.encoder = mit_b1(pretrained=pretrained)
            encoder_channels = [64, 128, 320, 512]
        elif phi == 'b2':
            self.encoder = mit_b2(pretrained=pretrained)
            encoder_channels = [64, 128, 320, 512]
        elif phi == 'b3':
            self.encoder = mit_b3(pretrained=pretrained)
            encoder_channels = [64, 128, 320, 512]
        elif phi == 'b4':
            self.encoder = mit_b4(pretrained=pretrained)
            encoder_channels = [64, 128, 320, 512]
        elif phi == 'b5':
            self.encoder = mit_b5(pretrained=pretrained)
            encoder_channels = [64, 128, 320, 512]
        else:
            raise ValueError(f"Unsupported MiT variant: {phi}")

        if mit_weights_path and not pretrained:
            print(f"Loading MiT backbone weights from (specific path): {mit_weights_path}")
            try:
                state_dict = torch.load(mit_weights_path, map_location='cpu')
                missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
                print(f"Loaded MiT backbone from {mit_weights_path}.")
                if missing_keys:
                    print("Missing keys in MiT backbone load:", missing_keys)
                if unexpected_keys:
                    print("Unexpected keys in MiT backbone load:", unexpected_keys)
            except Exception as e:
                print(f"Error loading MiT weights from {mit_weights_path}: {e}")
        elif pretrained:
            print(f"MiT backbone ({phi}) initialized requesting its internal pretrained weights.")

        self.embedding_dim = {'b0': 256, 'b1': 256, 'b2': 768, 'b3': 768, 'b4': 768, 'b5': 768}[phi]

        self.decode_head = SegformerHead(
            num_classes=num_classes,
            in_channels=encoder_channels,
            embedding_dim=self.embedding_dim,
            use_dcn_in_aspp=use_dcn_in_aspp,
            use_dasci_post_fusion=use_dasci_post_fusion,
            dasci_individual_mlp_stages=dasci_individual_mlp_stages,
            num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii
        )

        if self.use_gradient_checkpointing:
            def apply_checkpointing_to_block_list(block_list_module):
                if block_list_module is not None and isinstance(block_list_module, nn.ModuleList):
                    for i in range(len(block_list_module)):
                        if isinstance(block_list_module[i], nn.Module):
                            block_list_module[i] = CheckpointWrapper(block_list_module[i], enabled=True)
            apply_checkpointing_to_block_list(getattr(self.encoder, 'block1', None))
            apply_checkpointing_to_block_list(getattr(self.encoder, 'block2', None))
            apply_checkpointing_to_block_list(getattr(self.encoder, 'block3', None))
            apply_checkpointing_to_block_list(getattr(self.encoder, 'block4', None))

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        encoder_features = self.encoder.forward(inputs)
        seg, con0, con1 = self.decode_head.forward(encoder_features)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=True)
        con0 = F.interpolate(con0, size=(H, W), mode='bilinear', align_corners=True)
        con1 = F.interpolate(con1, size=(H, W), mode='bilinear', align_corners=True)
        if self.training:
            return seg, con0, con1
        else:
            return seg

# -------------------------------------------------------------
# CheckpointWrapper for Gradient Checkpointing
# -------------------------------------------------------------
class CheckpointWrapper(nn.Module):
    def __init__(self, module, enabled=True):
        super().__init__()
        self.module = module
        self.enabled = enabled

    def forward(self, *args):
        if self.enabled and self.module.training and torch.is_grad_enabled():
            if len(args) == 3 and isinstance(args[0], torch.Tensor) and isinstance(args[1], int) and isinstance(args[2], int):
                return torch.utils.checkpoint.checkpoint(self.module, args[0], args[1], args[2], use_reentrant=False)
            else:
                return torch.utils.checkpoint.checkpoint(self.module, *args, use_reentrant=False)
        else:
            return self.module(*args)

# -------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------
if __name__ == '__main__':
    print("--- Test Case 1: Default (DASCI post-fusion, no grouping in SPCII's 1D conv) ---")
    model_default = SegFormer(num_classes=2, phi='b0', use_dcn_in_aspp=True,
                              use_dasci_post_fusion=True,
                              dasci_individual_mlp_stages={'c1': False, 'c2': False, 'c3': False, 'c4': False},
                              num_groups_1d_conv_in_spcii=1)
    print(f"Post-fusion DASCI active: {isinstance(model_default.decode_head.post_fuse_dasci_module, DASCIModule)}")
    model_default.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    output_default = model_default(input_tensor)
    print(f"Output shape (eval mode): {output_default.shape}\n")

    print("--- Test Case 2: DASCI with grouping (e.g., 4 groups) in SPCII's 1D conv, post-fusion ---")
    model_grouped_spcii = SegFormer(num_classes=2, phi='b0', use_dcn_in_aspp=True,
                                    use_dasci_post_fusion=True,
                                    dasci_individual_mlp_stages={'c1': False, 'c2': False, 'c3': False, 'c4': False},
                                    num_groups_1d_conv_in_spcii=4)
    print(f"Post-fusion DASCI active: {isinstance(model_grouped_spcii.decode_head.post_fuse_dasci_module, DASCIModule)}")
    if isinstance(model_grouped_spcii.decode_head.post_fuse_dasci_module, DASCIModule):
        print(f"  SPCII num_groups for 1D conv: {model_grouped_spcii.decode_head.post_fuse_dasci_module.attention_module.part2_channel_embedded_space.num_groups_1d_conv}")
    model_grouped_spcii.eval()
    output_grouped = model_grouped_spcii(input_tensor)
    print(f"Output shape (eval mode, SPCII grouped): {output_grouped.shape}\n")

    print("--- Testing train mode output shapes (default DASCI) ---")
    model_default.train()
    seg_pred, con0_pred, con1_pred = model_default(input_tensor)
    print(f"Main Seg Prediction shape (train mode): {seg_pred.shape}")
    print(f"Con0 Prediction shape (train mode): {con0_pred.shape}")
    print(f"Con1 Prediction shape (train mode): {con1_pred.shape}\n")
