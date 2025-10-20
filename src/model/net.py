import torch
import torch.nn as nn
from .raft import RAFT
from .RAFT_component.raft_utils import load_ckpt
from .fusion import Fusion_Net
from .utils import load_args_from_json, flow_warp


class VideoFusion(nn.Module):
    def __init__(self, model_config):
        super(VideoFusion, self).__init__()
        self.raft_args = load_args_from_json("config/module/spring-S.json")
        self.flow_net = RAFT(self.raft_args).eval()
        load_ckpt(self.flow_net, self.raft_args.path)
        self.align_mode = model_config['model']['align_mode']
        self.output_mask = model_config['model']['output_mask']
        self.fusion_net = Fusion_Net(
            dim=model_config['model']['dim'], 
            num_blocks=model_config['model']['num_blocks'], 
            head=model_config['model']['head'], 
            ffn_expansion_factor=model_config['model']['ffn_expansion_factor'], 
            bias=model_config['model']['bias'], 
            LayerNorm_type=model_config['model']['LayerNorm_type'], 
            output_mask=model_config['model']['output_mask']
        )
        

    def extract_features(self, frames, index):
        """
        frames: [B, 5, 3, H, W] → [B*5, 3, H, W] → encoder → [B*5, C, H, W]
        Returns the reshaped tensor [B, 5, C, H, W]
        """
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        if index == 1:
            frames = self.fusion_net.encoder_1(frames)
        elif index == 2:
            frames = self.fusion_net.encoder_2(frames)
        return frames.view(B, T, -1, H, W)
        
    def prepare_window_features(self, feats, frames):
        """
        feats: [B, 5, C, H, W]
        frames: [B, 5, 3, H, W]
        Returns three [B, 3C, H, W] fusion inputs
        """
        B, T, C, H, W = feats.shape
        fused_features = []
        scaled_frames = frames * 255.0

        for i in range(3):  # Sliding windows: 0–2, 1–3, 2–4
            f_prev = feats[:, i]
            f_cur = feats[:, i + 1]
            f_nxt = feats[:, i + 2]

            I_prev = scaled_frames[:, i]
            I_cur = scaled_frames[:, i + 1]
            I_nxt = scaled_frames[:, i + 2]

            with torch.no_grad():
                flow_2_to_1 = self.flow_net(I_cur, I_prev)["final"]
                flow_2_to_3 = self.flow_net(I_cur, I_nxt)["final"]

            f_prev = flow_warp(f_prev, flow_2_to_1)
            f_nxt = flow_warp(f_nxt, flow_2_to_3)

            fused_input = torch.cat([f_prev, f_cur, f_nxt], dim=1)  # [B, 3C, H, W]
            fused_features.append(fused_input)

        return fused_features  # List of 3 × [B, 3C, H, W]

    def forward(self, sources_1, sources_2):
        """
        sources_1: [B, 5, 3, H, W]
        sources_2: [B, 5, 3, H, W]
        Output: [B, 3, 3, H, W]
        """
        # Extract features from both source sequences
        feat1 = self.extract_features(sources_1, index=1)  # [B, 5, 3, H, W] -> [B, 5, C, H, W]
        feat2 = self.extract_features(sources_2, index=2)  # [B, 5, 3, H, W] -> [B, 5, C, H, W]

        # Prepare temporal window features for fusion
        feat1 = self.prepare_window_features(feat1, sources_1)
        feat2 = self.prepare_window_features(feat2, sources_2)

        # Frame-level fusion
        outputs = []
        for i in range(3):
            out = self.fusion_net(feat1[i], feat2[i])  # [B, 3, H, W]
            if self.output_mask:
                out = sources_1[:, i + 1, :, :, :] * out + sources_2[:, i + 1, :, :, :] * (1 - out)
            outputs.append(out.unsqueeze(1))  # [B, 1, 3, H, W]

        return torch.cat(outputs, dim=1), self.flow_net  # [B, 3, 3, H, W], flow_net