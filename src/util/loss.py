# Last modified: 2025-10-20
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss
from torch.autograd import Variable
from math import exp
def rgb2ycrcb(rgb_tensor):
    r = rgb_tensor[:, 0, :, :]
    g = rgb_tensor[:, 1, :, :]
    b = rgb_tensor[:, 2, :, :]
    
    # Conversion formula (ITU-R BT.601 standard)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5  # Typically Cr range [0,1]
    cb = (b - y) * 0.564 + 0.5  # Typically Cb range [0,1]
    
    # Stack channels
    ycrcb_tensor = torch.stack([y, cr, cb], dim=1)
    
    # Clamp values to valid range
    ycrcb_tensor = torch.clamp(ycrcb_tensor, 0.0, 1.0)
    
    return ycrcb_tensor

def get_loss(loss_name, **kwargs):
    if loss_name == "IVF_MVF_Loss":
        return IVF_MVF_Loss(**kwargs)
    elif loss_name == "MEF_Loss":
        return MEF_Loss(**kwargs)
    elif loss_name == "MFF_Loss":
        return MFF_Loss(**kwargs)   
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # Compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # Compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2,
                         groups=C).view(C, H, W) - muX_sq

    # Compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muX.expand_as(muY_seq) * muY_seq

    # Compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0,
                          patch_index.view(1, -1)).view(C, H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()
    return q


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        # channel here should be the channel of the *fused* image X
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        # Register window as a buffer so it's moved with the module and saved in state_dict
        self.register_buffer('window', create_window(window_size, self.channel))
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        # X: fused image batch (BxCxHxW)
        # Ys: source images batch (BxKxCxHxW), where K is the number of source images per item (e.g., 2 for IR/VIS)

        B, C, H, W = X.size() # Get batch size and fused image dimensions
        K = Ys.size(1) # Get the number of source images per item
        channel_fused = C # The number of channels in the fused image

        # Ensure the window matches the fused image's channel and device/type
        # If the channel count changes or device/type mismatch, recreate the window
        if channel_fused != self.channel or self.window.device != X.device or self.window.dtype != X.dtype:
             window = create_window(self.window_size, channel_fused).to(X.device).type(X.dtype)
             self.register_buffer('window', window) # Update the registered buffer
             self.channel = channel_fused # Update the stored channel count
        else:
             window = self.window # Use the existing registered buffer

        ssim_scores = []
        # Loop through each item in the batch
        for i in range(B):
            # Extract one fused image: shape becomes 1xCxHxW
            x_i = X[i].unsqueeze(0)
            # Extract the set of source images for this item: shape is KxCxHxW
            ys_i = Ys[i]

            # Call the actual _mef_ssim function with single fused image and its source set
            # Your _mef_ssim function MUST be defined to accept these shapes: X: 1xCxHxW, Ys: KxCxHxW
            score = _mef_ssim(x_i, ys_i, window, self.window_size,
                              self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)
            ssim_scores.append(score)

        # Based on your loss formula (batch_size - sum(scores)), we return the sum of scores over the batch
        total_ssim_score = torch.mean(torch.stack(ssim_scores))
        return total_ssim_score # This value will be ssimscore in your training loop
    
class MEF_Loss(nn.Module):
    def __init__(
        self,
        coef,
        use_occlusion,
        occ_threshold,
        max_offset_mask
    ):
        super().__init__()
        self.coef = tuple(coef)
        self.mefssim=MEFSSIM(channel=1, is_lum=False)
        self.kernelx = (
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernely = (
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.use_occlusion = use_occlusion
        self.occ_threshold = occ_threshold
        self.max_offset_mask = max_offset_mask
        from src.model.utils import flow_warp
        self.flow_warp = flow_warp

    def sobel_filter(self, tensor):
        # Compute Sobel gradient
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Expand to match channels
        sobel_x = sobel_x.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)
        sobel_y = sobel_y.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)

        grad_x = F.conv2d(tensor, sobel_x, padding=1, groups=tensor.shape[1])
        grad_y = F.conv2d(tensor, sobel_y, padding=1, groups=tensor.shape[1])

        grad_x = torch.mean(torch.abs(grad_x), dim=1, keepdim=True)
        grad_y = torch.mean(torch.abs(grad_y), dim=1, keepdim=True)

        return grad_x + grad_y

    def compute_single_loss(self, f, a, b):
        f_ycrcb=rgb2ycrcb(f)
        a_ycrcb=rgb2ycrcb(a)
        b_ycrcb=rgb2ycrcb(b)

        grad_f = self.sobel_filter(f_ycrcb[:,:1,:,:])
        grad_a = self.sobel_filter(a_ycrcb[:,:1,:,:])
        grad_b = self.sobel_filter(b_ycrcb[:,:1,:,:])

        loss_int = F.l1_loss(f_ycrcb[:,:1,:,:], (a_ycrcb[:,:1,:,:] + b_ycrcb[:,:1,:,:]) / 2)
        loss_grad = F.l1_loss(grad_f, torch.max(grad_a, grad_b))

        CrCb_fuse=(a_ycrcb[:,1:,:,:]*torch.abs(a_ycrcb[:,1:,:,:]-0.5)+b_ycrcb[:,1:,:,:]*torch.abs(b_ycrcb[:,1:,:,:]-0.5))/(torch.abs(a_ycrcb[:,1:,:,:]-0.5)+torch.abs(b_ycrcb[:,1:,:,:]-0.5)+1e-8)

        loss_color= F.l1_loss(f_ycrcb[:,1:,:,:], CrCb_fuse)
        loss_ssim= 1-self.mefssim(f_ycrcb,torch.stack([a_ycrcb, b_ycrcb], dim=1))
        return loss_int, loss_grad, loss_color, loss_ssim

    def spatial_loss(self, f, a, b): 
        """
        f: [B, 3, C, H, W]
        a: [B, 5, C, H, W]
        b: [B, 5, C, H, W]
        """
        total_loss_int = 0.0
        total_loss_grad = 0.0
        total_loss_color=0.0
        total_loss_ssim=0.0
        for i in range(3):
            fi = f[:, i, :, :, :]
            ai = a[:, i + 1, :, :, :]
            bi = b[:, i + 1, :, :, :]
            loss_int, loss_grad, loss_color, loss_ssim = self.compute_single_loss(fi, ai, bi)
            total_loss_int += loss_int
            total_loss_grad += loss_grad
            total_loss_color+= loss_color
            total_loss_ssim+= loss_ssim
        return total_loss_int, total_loss_grad,total_loss_color,total_loss_ssim

    @torch.no_grad()
    def occlusion_mask(self, img1, img2, flow_ab, flow_net):
        """Compute forward-backward consistency occlusion mask."""
        flow_ba = flow_net(img2, img1)["final"]
        flow_ba_warped = self.flow_warp(flow_ba, flow_ab)
        fb_diff = flow_ab + flow_ba_warped
        fb_consistency = fb_diff.norm(p=2, dim=1)  # [B, H, W]
        mask = (fb_consistency < self.occ_threshold).float()
        return mask

    def temporal_loss(self, f, a, b, flow_net):
        f_prev, f_cur, f_nxt= f[:, 0]*255, f[:, 1]*255, f[:, 2]*255
        flow_net.eval()

        with torch.no_grad():
            flow_f_prev2cur = flow_net(f_prev, f_cur)["final"]
            flow_f_next2cur = flow_net(f_nxt, f_cur)["final"]
        f_cur, f_prev, f_nxt= f_cur/255, f_prev/255, f_nxt/255

        
        f_recon_prev = self.flow_warp(f_prev, flow_f_prev2cur)
        f_recon_next = self.flow_warp(f_nxt, flow_f_next2cur)



        if self.use_occlusion:
            mask_prev = self.occlusion_mask(f_prev, f_cur, flow_f_prev2cur, flow_net)
            mask_next = self.occlusion_mask(f_nxt, f_cur, flow_f_next2cur, flow_net)
        else:
            mask_prev = torch.ones(
                (f.shape[0], f.shape[3], f.shape[4]), device=f.device
            )
            mask_next = torch.ones(
                (f.shape[0], f.shape[3], f.shape[4]), device=f.device
            )

        if self.max_offset_mask:
            mask_prev[torch.max(torch.abs(flow_f_prev2cur),1)[0]>3]  =0  
            mask_next[torch.max(torch.abs(flow_f_next2cur),1)[0]>3]  =0

        spatial_diff_prev = (torch.abs(f_cur - f_recon_prev)).mean(1)
        spatial_diff_next = (torch.abs(f_cur - f_recon_next)).mean(1)

        spatial_err_prev = (mask_prev * spatial_diff_prev).sum(dim=(1, 2)) / (
            mask_prev.sum(dim=(1, 2)) + 1e-10
        )
        spatial_err_next = (mask_next * spatial_diff_next).sum(dim=(1, 2)) / (
            mask_next.sum(dim=(1, 2)) + 1e-10
        )
        return (spatial_err_prev + spatial_err_next).mean() 

    def forward(self, f, a, b, flow_net):
        loss_int, loss_grad, loss_color, loss_ssim= self.spatial_loss(f, a, b)
        loss_temp = self.temporal_loss(f, a, b, flow_net)
        
        total_loss = (
            self.coef[0] * (loss_int + loss_color)
            + self.coef[1] * loss_grad
            + self.coef[2] * loss_ssim
            + self.coef[3] * loss_temp
        )
        loss_output = {
            "loss": total_loss,
            "loss_int": loss_int,
            "loss_grad": loss_grad,
            "loss_ssim": loss_ssim,
            "loss_temp": loss_temp,
        }
        return loss_output

    
class IVF_MVF_Loss(nn.Module):
    def __init__(self, coef, use_occlusion, occ_threshold, max_offset_mask):
        super().__init__()
        self.coef = tuple(coef)
        self.kernelx = (
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernely = (
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.use_occlusion = use_occlusion
        self.occ_threshold = occ_threshold
        self.max_offset_mask = max_offset_mask
        from src.model.utils import flow_warp

        self.flow_warp = flow_warp

    def sobel_filter(self, tensor):
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sobel_x = sobel_x.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)
        sobel_y = sobel_y.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)

        grad_x = F.conv2d(tensor, sobel_x, padding=1, groups=tensor.shape[1])
        grad_y = F.conv2d(tensor, sobel_y, padding=1, groups=tensor.shape[1])

        grad_x = torch.mean(torch.abs(grad_x), dim=1, keepdim=True)
        grad_y = torch.mean(torch.abs(grad_y), dim=1, keepdim=True)

        return grad_x + grad_y

    def compute_single_loss(self, f, a, b):

        f_ycrcb=rgb2ycrcb(f)
        a_ycrcb=rgb2ycrcb(a)
        b_ycrcb=rgb2ycrcb(b)

        grad_f = self.sobel_filter(f_ycrcb[:,:1,:,:])
        grad_a = self.sobel_filter(a_ycrcb[:,:1,:,:])
        grad_b = self.sobel_filter(b_ycrcb[:,:1,:,:])

        loss_int = F.l1_loss(f_ycrcb[:,:1,:,:],torch.max(a_ycrcb[:,:1,:,:],b_ycrcb[:,:1,:,:]))
        loss_grad = F.l1_loss(grad_f, torch.max(grad_a, grad_b))
        loss_color= F.l1_loss(f_ycrcb[:,1:,:,:],b_ycrcb[:,1:,:,:])
        loss_ssim= 0.5*SSIMLoss(11, reduction='mean')(f,a)+0.5*SSIMLoss(11, reduction='mean')(f,b)

        return loss_int, loss_grad, loss_color, loss_ssim

    def spatial_loss(self, f, a, b): 
        """
        f: [B, 3, C, H, W]
        a: [B, 5, C, H, W]
        b: [B, 5, C, H, W]
        """
        total_loss_int = 0.0
        total_loss_grad = 0.0
        total_loss_color = 0.0
        total_loss_ssim = 0.0
        for i in range(3):
            fi = f[:, i, :, :, :]
            ai = a[:, i + 1, :, :, :]
            bi = b[:, i + 1, :, :, :]
            loss_int, loss_grad, loss_color, loss_ssim = self.compute_single_loss(
                fi, ai, bi
            )
            total_loss_int += loss_int
            total_loss_grad += loss_grad
            total_loss_color += loss_color
            total_loss_ssim += loss_ssim
        return total_loss_int, total_loss_grad, total_loss_color, total_loss_ssim

    @torch.no_grad()
    def occlusion_mask(self, img1, img2, flow_ab, flow_net):
        """Compute forward-backward consistency occlusion mask."""
        flow_ba = flow_net(img2, img1)["final"]
        flow_ba_warped = self.flow_warp(flow_ba, flow_ab)
        fb_diff = flow_ab + flow_ba_warped
        fb_consistency = fb_diff.norm(p=2, dim=1)  # [B, H, W]
        mask = (fb_consistency < self.occ_threshold).float()
        return mask

    def temporal_loss(self, f, a, b, flow_net):
        f_prev, f_cur, f_nxt = f[:, 0] * 255, f[:, 1] * 255, f[:, 2] * 255
        flow_net.eval()

        with torch.no_grad():
            flow_f_prev2cur = flow_net(f_prev, f_cur)["final"]
            flow_f_next2cur = flow_net(f_nxt, f_cur)["final"]

        f_cur, f_prev, f_nxt = f_cur / 255, f_prev / 255, f_nxt / 255

        f_recon_prev = self.flow_warp(f_prev, flow_f_prev2cur)
        f_recon_next = self.flow_warp(f_nxt, flow_f_next2cur)

        if self.use_occlusion:
            mask_prev = self.occlusion_mask(f_prev, f_cur, flow_f_prev2cur, flow_net)
            mask_next = self.occlusion_mask(f_nxt, f_cur, flow_f_next2cur, flow_net)
        else:
            mask_prev = torch.ones(
                (f.shape[0], f.shape[3], f.shape[4]), device=f.device
            )
            mask_next = torch.ones(
                (f.shape[0], f.shape[3], f.shape[4]), device=f.device
            )

        if self.max_offset_mask:
            mask_prev[torch.max(torch.abs(flow_f_prev2cur), 1)[0] > 3] = 0
            mask_next[torch.max(torch.abs(flow_f_next2cur), 1)[0] > 3] = 0

        spatial_diff_prev = (torch.abs(f_cur - f_recon_prev)).mean(1)
        spatial_diff_next = (torch.abs(f_cur - f_recon_next)).mean(1)

        spatial_err_prev = (mask_prev * spatial_diff_prev).sum(dim=(1, 2)) / (
            mask_prev.sum(dim=(1, 2)) + 1e-10
        )
        spatial_err_next = (mask_next * spatial_diff_next).sum(dim=(1, 2)) / (
            mask_next.sum(dim=(1, 2)) + 1e-10
        )
        return (spatial_err_prev + spatial_err_next).mean()

    def forward(self, f, a, b, flow_net):
        loss_int, loss_grad, loss_color, loss_ssim = self.spatial_loss(f, a, b)
        loss_temp = self.temporal_loss(f, a, b, flow_net)

        total_loss = (
            self.coef[0] * (loss_int + loss_color)
            + self.coef[1] * loss_grad
            + self.coef[2] * loss_ssim
            + self.coef[3] * loss_temp
        )
        loss_output = {
            "loss": total_loss,
            "loss_int": loss_int,
            "loss_grad": loss_grad,
            "loss_ssim": loss_ssim,
            "loss_temp": loss_temp,
        }
        return loss_output


class MFF_Loss(nn.Module):
    def __init__(
        self,
        coef,
        use_occlusion,
        occ_threshold,
        max_offset_mask
    ):
        super().__init__()
        self.coef = tuple(coef)
        self.kernelx = (
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernely = (
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.use_occlusion = use_occlusion
        self.occ_threshold = occ_threshold
        self.max_offset_mask = max_offset_mask
        from src.model.utils import flow_warp
        self.flow_warp = flow_warp

    def sobel_filter(self, tensor):
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sobel_x = sobel_x.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)
        sobel_y = sobel_y.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)

        grad_x = F.conv2d(tensor, sobel_x, padding=1, groups=tensor.shape[1])
        grad_y = F.conv2d(tensor, sobel_y, padding=1, groups=tensor.shape[1])

        grad_x = torch.mean(torch.abs(grad_x), dim=1, keepdim=True)
        grad_y = torch.mean(torch.abs(grad_y), dim=1, keepdim=True)

        return grad_x + grad_y

    def compute_single_loss(self, f, a, b):
        f_ycrcb=rgb2ycrcb(f)
        a_ycrcb=rgb2ycrcb(a)
        b_ycrcb=rgb2ycrcb(b)

        grad_f = self.sobel_filter(f_ycrcb[:,:1,:,:])
        grad_a = self.sobel_filter(a_ycrcb[:,:1,:,:])
        grad_b = self.sobel_filter(b_ycrcb[:,:1,:,:])

        loss_int = F.l1_loss(f_ycrcb[:,:1,:,:], (a_ycrcb[:,:1,:,:] + b_ycrcb[:,:1,:,:]) / 2)
        loss_grad = F.l1_loss(grad_f, torch.max(grad_a, grad_b))
        CrCb_fuse=(a_ycrcb[:,1:,:,:]*torch.abs(a_ycrcb[:,1:,:,:]-0.5)+b_ycrcb[:,1:,:,:]*torch.abs(b_ycrcb[:,1:,:,:]-0.5))/(torch.abs(a_ycrcb[:,1:,:,:]-0.5)+torch.abs(b_ycrcb[:,1:,:,:]-0.5)+1e-8)
        loss_color= F.l1_loss(f_ycrcb[:,1:,:,:], CrCb_fuse)

        loss_ssim= 0.5*SSIMLoss(11, reduction='mean')(f,a)+0.5*SSIMLoss(11, reduction='mean')(f,b)
        return loss_int, loss_grad, loss_color, loss_ssim

    def spatial_loss(self, f, a, b): 
        """
        f: [B, 3, C, H, W]
        a: [B, 5, C, H, W]
        b: [B, 5, C, H, W]
        """
        total_loss_int = 0.0
        total_loss_grad = 0.0
        total_loss_color=0.0
        total_loss_ssim=0.0
        for i in range(3):
            fi = f[:, i, :, :, :]
            ai = a[:, i + 1, :, :, :]
            bi = b[:, i + 1, :, :, :]
            loss_int, loss_grad, loss_color, loss_ssim = self.compute_single_loss(fi, ai, bi)
            total_loss_int += loss_int
            total_loss_grad += loss_grad
            total_loss_color+= loss_color
            total_loss_ssim+= loss_ssim
        return total_loss_int, total_loss_grad,total_loss_color,total_loss_ssim

    @torch.no_grad()
    def occlusion_mask(self, img1, img2, flow_ab, flow_net):
        """Compute forward-backward consistency occlusion mask."""
        flow_ba = flow_net(img2, img1)["final"]
        flow_ba_warped = self.flow_warp(flow_ba, flow_ab)
        fb_diff = flow_ab + flow_ba_warped
        fb_consistency = fb_diff.norm(p=2, dim=1)  # [B, H, W]
        mask = (fb_consistency < self.occ_threshold).float()
        return mask

    def temporal_loss(self, f, a, b, flow_net):
        f_prev, f_cur, f_nxt= f[:, 0]*255, f[:, 1]*255, f[:, 2]*255
        flow_net.eval()

        with torch.no_grad():
            flow_f_prev2cur = flow_net(f_prev, f_cur)["final"]
            flow_f_next2cur = flow_net(f_nxt, f_cur)["final"]


        f_cur, f_prev, f_nxt= f_cur/255, f_prev/255, f_nxt/255

        
        f_recon_prev = self.flow_warp(f_prev, flow_f_prev2cur)
        f_recon_next = self.flow_warp(f_nxt, flow_f_next2cur)



        if self.use_occlusion:
            mask_prev = self.occlusion_mask(f_prev, f_cur, flow_f_prev2cur, flow_net)
            mask_next = self.occlusion_mask(f_nxt, f_cur, flow_f_next2cur, flow_net)
        else:
            mask_prev = torch.ones(
                (f.shape[0], f.shape[3], f.shape[4]), device=f.device
            )
            mask_next = torch.ones(
                (f.shape[0], f.shape[3], f.shape[4]), device=f.device
            )

        if self.max_offset_mask:
            mask_prev[torch.max(torch.abs(flow_f_prev2cur),1)[0]>3]  =0  
            mask_next[torch.max(torch.abs(flow_f_next2cur),1)[0]>3]  =0

        spatial_diff_prev = (torch.abs(f_cur - f_recon_prev)).mean(1)
        spatial_diff_next = (torch.abs(f_cur - f_recon_next)).mean(1)

        spatial_err_prev = (mask_prev * spatial_diff_prev).sum(dim=(1, 2)) / (
            mask_prev.sum(dim=(1, 2)) + 1e-10
        )
        spatial_err_next = (mask_next * spatial_diff_next).sum(dim=(1, 2)) / (
            mask_next.sum(dim=(1, 2)) + 1e-10
        )
        return (spatial_err_prev + spatial_err_next).mean()

    def forward(self, f, a, b, flow_net):
        loss_int, loss_grad, loss_color, loss_ssim= self.spatial_loss(f, a, b)
        loss_temp = self.temporal_loss(f, a, b, flow_net)
        
        total_loss = (
            self.coef[0] * (loss_int + loss_color)
            + self.coef[1] * loss_grad
            + self.coef[2] * loss_ssim
            + self.coef[3] * loss_temp
        )
        loss_output = {
            "loss": total_loss,
            "loss_int": loss_int,
            "loss_grad": loss_grad,
            "loss_ssim": loss_ssim,
            "loss_temp": loss_temp,
        }
        return loss_output  