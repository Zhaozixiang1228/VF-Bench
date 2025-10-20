# Last modified: 2025-10-17


import pandas as pd
import torch
import torch.nn.functional as F
import torch.fft
import inspect

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def _input_check(batch, ref1=None, ref2=None):  # Check input
    def _batch_check(batch):    
        assert type(batch) is torch.Tensor, "input is not a tensor"

        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)

        assert len(batch.shape) == 4, "dimension number error"  
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        
        assert batch.shape[1] == 3, "channel number error"
        # For some medical images, the batch can be completely black
        if torch.max(batch) < 1:
            noise = torch.randn_like(batch) * 0.1 + 1  # Add random noise with mean 1, or metric calculation will have bugs
            batch = batch + noise
        return batch
    
    assert batch.device.type != 'cpu', "input is on CPU"
    batch = _batch_check(batch).to(torch.float64)
    if ref1 is not None:
        ref1 = _batch_check(ref1).to(torch.float64).to(batch.device)
    if ref2 is not None:
        ref2 = _batch_check(ref2).to(torch.float64).to(batch.device)

    return batch, ref1, ref2


def Metric_VIF(batch, ref1=None, ref2=None):
    def _vifp_batch(ref, dist):
        ref = ref.to(torch.float64)
        dist = dist.to(torch.float64)
        sigma_nsq = 2.0  # Visual noise variance
        eps = 1e-10
        num = torch.zeros(ref.shape[0], ref.shape[1]).to(dist.device)
        den = torch.zeros(ref.shape[0], ref.shape[1]).to(dist.device)

        for scale in range(1, 5):  # 4 scales
            N = 2**(4 - scale + 1) + 1
            sd = N / 5.0
            win = _fspecial_gaussian_torch(N, sd).to(dist.device)

            if scale > 1:
                ref = F.conv2d(ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])[:, :, ::2, ::2]
                dist = F.conv2d(dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])[:, :, ::2, ::2]

            mu1 = F.conv2d(ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])
            mu2 = F.conv2d(dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(ref * ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1]) - mu1_sq
            sigma2_sq = F.conv2d(dist * dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1]) - mu2_sq
            sigma12 = F.conv2d(ref * dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1]) - mu1_mu2
            sigma1_sq = torch.clamp(sigma1_sq, min=0)
            sigma2_sq = torch.clamp(sigma2_sq, min=0)
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)), dim=(2,3))
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=(2,3))

        vifp_val = num / den
        vifp_val = _ignore_nan(vifp_val)
        return vifp_val
    
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        vifp1 = torch.mean(_vifp_batch(ref1, batch), dim=1)  # (B,)
        vifp2 = torch.mean(_vifp_batch(ref2, batch), dim=1)  # (B,)
        return (vifp1 + vifp2) / 2.0 

def Metric_MEF_SSIM(batch, ref1=None, ref2=None):
    def mef_ssim(img, img1, img2, K=0.03, window=None):
        imgSeq = torch.cat((img1.unsqueeze(-1), img2.unsqueeze(-1)), dim=-1)  #B,C,H,W,2

        if window is None:
            window= _fspecial_gaussian_torch(11, 1.5).to(img.device)

        wSize = window.shape[2]
        sWindow = torch.ones((wSize, wSize), device=img.device,dtype=torch.float64) / wSize**2
        bd = wSize // 2 

        # Calculate mu and ed (mean and standard deviation in the local window)
        imgSeq=torch.cat((imgSeq[:,:,:,:,0], imgSeq[:,:,:,:,1]), dim=1) # B,2*C,H,W
        mu=F.conv2d(imgSeq, sWindow.repeat(imgSeq.shape[1],1,1,1), padding=0,groups=imgSeq.shape[1]) # B,2*C,H,W

        sigmaSq= F.conv2d(imgSeq**2, sWindow.repeat(imgSeq.shape[1],1,1,1), padding=0,groups=imgSeq.shape[1])-mu** 2 # B,2*C,H,W

        
        ed=torch.sqrt(torch.relu(wSize*wSize*sigmaSq)) + 0.001 # B,2*C,H,W
        del sigmaSq
        temp1,temp2=torch.chunk(ed.unsqueeze(-1), 2, dim=1)
        ed=torch.cat([temp1,temp2],dim=-1) # B,C,H,W,2
        del temp1,temp2
        # Consistency map R
        vecs = imgSeq.unfold(2, 2*bd+1, 1).unfold(3, 2*bd+1, 1) # B,2*C,H,W,2*bd+1,2*bd+1
        vecs =vecs.flatten(start_dim=4) # B,2*C,H,W,(2*bd+1)*(2*bd+1)
        denominator= torch.norm(vecs-mu.unsqueeze(-1), p=2, dim=4) # B,2*C,H,W
        temp1,temp2=torch.chunk(denominator, 2, dim=1)
        denominator=temp1+temp2

        temp1,temp2=torch.chunk(vecs.unsqueeze(-1), 2, dim=1)
        
        vecs=torch.cat([temp1,temp2],dim=-1) # B,C,H,W,(2*bd+1)*(2*bd+1),2
        del temp1,temp2
        vecs_sum = torch.sum(vecs,dim=-1)
        del vecs  # B,C,H,W,(2*bd+1)*(2*bd+1)
        numerator = torch.norm(vecs_sum - torch.mean(vecs_sum,dim=-1,keepdim=True), p=2, dim=-1)
        del vecs_sum
        R=(numerator + torch.finfo(torch.float64).eps) / (denominator + torch.finfo(torch.float64).eps)
        del numerator, denominator

        R = torch.clamp(R, min=torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
        p = torch.tan(torch.pi / 2 * R)
        p = torch.clamp(p, min=0, max=10)

        wMap = (ed / wSize) ** (p.unsqueeze(-1)) + torch.finfo(torch.float64).eps # B,C,H,W,2
        wMap /= torch.sum(wMap, dim=-1,keepdim=True) # B,C,H,W,2

        maxEd = torch.max(ed, dim=-1)[0] # B,C,H,W
        
        C = (K * 255) ** 2
        blocks = imgSeq.unfold(2, 2*bd+1, 1).unfold(3, 2*bd+1, 1) # B,2*C,H,W,2*bd+1,2*bd+1
        blocks =blocks.flatten(start_dim=4) # B,2*C,H,W,(2*bd+1)*(2*bd+1)
        temp1,temp2=torch.chunk(blocks.unsqueeze(-1), 2, dim=1)
        blocks=torch.cat([temp1,temp2],dim=-1) # B,C,H,W,(2*bd+1)*(2*bd+1),2

        temp1,temp2=torch.chunk(mu.unsqueeze(-1), 2, dim=1)
        mu=torch.cat([temp1,temp2],dim=-1) # B,C,H,W,2

        rBlock=wMap.unsqueeze(4)*(blocks-mu.unsqueeze(4))/ed.unsqueeze(4)  # B,C,H,W,(2*bd+1)*(2*bd+1),2

        del ed,wMap,blocks,temp1,temp2,mu

        rBlock=rBlock.sum(-1)
        temp1=torch.norm(rBlock, p=2, dim=4,keepdim=True) # index of norm>0 # B,C,H,W,1
        rBlock=(temp1!=0)*rBlock/(temp1+(temp1==0))*maxEd.unsqueeze(-1) # B,C,H,W,(2*bd+1)*(2*bd+1)
        del maxEd,temp1
        fBlock = img.unfold(2, 2*bd+1, 1).unfold(3, 2*bd+1, 1).flatten(start_dim=4) # B,C,H,W,(2*bd+1)*(2*bd+1)

        window=window.flatten().view(1, 1, 1, 1, -1)

        mu1=torch.sum(window*rBlock,4,keepdim=True) # B,C,H,W,1
        mu2=torch.sum(window*fBlock,4,keepdim=True) # B,C,H,W,1
        sigma1Sq = torch.sum(window * (rBlock - mu1) ** 2,dim=-1) # B,C,H,W
        sigma2Sq = torch.sum(window * (fBlock - mu2) ** 2,dim=-1) # B,C,H,W
        sigma12=torch.sum(window * (rBlock - mu1) *(fBlock - mu2),dim=-1) # B,C,H,W
        del rBlock,fBlock,mu1,mu2
        qMap= (2 * sigma12 + C) / (sigma1Sq + sigma2Sq + C)
        Q=torch.mean(qMap,dim=[2,3])


        return Q
    
    with torch.no_grad():
        batch,ref1,ref2=_input_check(batch,ref1,ref2)
        batch,ref1,ref2=rgb2gray(batch),rgb2gray(ref1),rgb2gray(ref2)

        return torch.mean(mef_ssim(batch,ref1,ref2),dim=1)

def Metric_SSIM(batch, ref1=None, ref2=None):
    def filter2D(img,win):
        img=F.pad(img, pad=(win.shape[-2]//2,win.shape[-1]//2,win.shape[-2]//2,win.shape[-1]//2), mode='reflect')

        return F.conv2d(img, win.repeat(img.shape[1],1,1,1), padding=0, groups=img.shape[1])

    def ssim(img, img_ref):
        K1=(0.01*255)**2
        K2=(0.03*255)**2
        sigma=1.5
        window_size=11

        win = _fspecial_gaussian_torch(window_size, sigma).to(img.device)


        mu1=filter2D(img, win)
        mu2=filter2D(img_ref, win)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = filter2D(img* img, win) - mu1_sq
        sigma2_sq = filter2D(img_ref * img_ref, win) - mu2_sq
        sigma12 = filter2D(img * img_ref, win) - mu1_mu2

        ssim_map = ((2. * mu1_mu2 + K1) * (2. * sigma12 + K2)) / \
        ((mu1_sq + mu2_sq + K1) * (sigma1_sq + sigma2_sq + K2))

        return torch.mean(ssim_map,dim=(2,3))
    
    with torch.no_grad():
        batch,ref1,ref2=_input_check(batch,ref1,ref2)
        output=(ssim(batch,ref1)+ssim(batch,ref2))/2
        return torch.mean(output,dim=1)
    
def Metric_MI(batch, ref1=None, ref2=None):
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        _, mi1 = _torch_normalized_mutual_info_score(batch, ref1) 
        _, mi2 = _torch_normalized_mutual_info_score(batch, ref2) 
    
    return torch.mean(mi1, dim=1) + torch.mean(mi2, dim=1)


def Metric_Qabf(batch, ref1, ref2):
    def Qabf_getArray(img):
        h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64, device=img.device).view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64, device=img.device).view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        
        SAx = F.conv2d(img, h3, padding=1, groups=img.shape[1])
        SAy = F.conv2d(img, h1, padding=1, groups=img.shape[1])
        
        gA = torch.sqrt(SAx.pow(2) + SAy.pow(2))
        aA = torch.zeros_like(img)
        aA[SAx == 0] = torch.pi / 2  # Handle case where SAx == 0
        mask = SAx != 0  # Boolean mask for nonzero elements in SAx
        aA[mask] = torch.atan(SAy[mask] / SAx[mask])
        return gA, aA

    def Qabf_getQabf(aA, gA, aF, gF):
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8
        
        GAF = torch.where(gA > gF, gF / gA,
                        torch.where(gA < gF, gA / gF,
                                    gF))

        AAF = 1 - (torch.abs(aA - aF) / (torch.pi / 2))
        
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
        QAF = QgAF * QaAF
        
        return QAF

    def Qabf(img, img1, img2):
        gA, aA = Qabf_getArray(img1)
        gB, aB = Qabf_getArray(img2)
        gF, aF = Qabf_getArray(img)
        
        QAF = Qabf_getQabf(aA, gA, aF, gF)
        QBF = Qabf_getQabf(aB, gB, aF, gF)
        
        nume = torch.sum(QAF * gA + QBF * gB, dim=(2, 3))
        deno = torch.sum(gA + gB, dim=(2, 3))
        
        return nume / deno
    
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        results = Qabf(batch, ref1, ref2) # [B, C]
        return torch.mean(results, dim=1) # [B,]
    
def _torch_normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'):
    """
    PyTorch implementation of NMI, fully aligned with sklearn.metrics.normalized_mutual_info_score
    Input shape: (B, H*W) or broadcastable to this shape
    """
    batch_size = labels_true.shape[0]
    nmi_values = torch.zeros(batch_size, dtype=torch.float64)
    mi_calculate = torch.zeros(labels_true.shape[0], labels_true.shape[1])
    nmi_calculate = torch.zeros(labels_true.shape[0], labels_true.shape[1])
    for b in range(batch_size):
        for c in range(labels_true.shape[1]):
            lbl_true = labels_true[b, c].flatten()
            lbl_pred = labels_pred[b, c].flatten()
            
            # Handle special case when all labels are identical (sklearn logic)
            unique_true = torch.unique(lbl_true)
            unique_pred = torch.unique(lbl_pred)
            if (len(unique_true) == len(unique_pred) == 1) or (len(unique_true) == len(unique_pred) == 0):
                nmi_values[b] = 1.0
                continue
            
            # Compute contingency matrix
            contingency = _torch_contingency_matrix(lbl_true, lbl_pred)
            contingency = contingency.to(torch.float64)
            total = contingency.sum()
            if total == 0:
                nmi_values[b] = 1.0
                continue
            
            # Compute mutual information MI (same formula as sklearn)
            pi = contingency.sum(dim=1)
            pj = contingency.sum(dim=0)
            log_contingency = (contingency / total).log()
            log_pi = (pi / total).log().unsqueeze(1)
            log_pj = (pj / total).log().unsqueeze(0)
            
            mi = (contingency / total) * (log_contingency - log_pi - log_pj)
            mi = mi.nansum()  # handle 0*log0
            mi_calculate[b,c] = mi
            
            # Handle mi=0 case (sklearn logic)
            if mi <= 1e-15:
                nmi_values[b] = 0.0
                continue
            
            # Compute entropy using natural log
            h_true = (-(pi[pi > 0] / pi.sum()).log() * (pi[pi > 0] / pi.sum())).sum()
            h_pred = (-(pj[pj > 0] / pj.sum()).log() * (pj[pj > 0] / pj.sum())).sum()
            
            # Select normalization method
            if average_method == 'arithmetic':
                normalizer = 0.5 * (h_true + h_pred)
            elif average_method == 'geometric':
                normalizer = torch.sqrt(h_true * h_pred)
            else:
                raise ValueError(f"Unsupported average_method: {average_method}")
            
            nmi_values[b] = mi / normalizer
            nmi_calculate[b,c] = nmi_values[b]
    return nmi_calculate, mi_calculate
    
def _torch_contingency_matrix(labels_true, labels_pred):
    """PyTorch implementation of contingency matrix, aligned with sklearn"""
    unique_true, map_true = torch.unique(labels_true, return_inverse=True)
    unique_pred, map_pred = torch.unique(labels_pred, return_inverse=True)
    
    contingency = torch.zeros(
        (len(unique_true), len(unique_pred)),  
        dtype=torch.int64,  
        device=labels_true.device
    )
    contingency.index_put_(
        (map_true, map_pred),  
        torch.ones_like(map_true, dtype=torch.int64),  
        accumulate=True
    )
    return contingency

def _normalize(tensor):
    """Normalize tensor to [0, 255] along the last two dimensions"""
    min_val = torch.min(tensor.flatten(start_dim=2), dim=2)[0]
    max_val = torch.max(tensor.flatten(start_dim=2), dim=2)[0]
    normalized = torch.zeros_like(tensor)
    mask = (max_val - min_val) != 0
    min_val = min_val.unsqueeze(-1).unsqueeze(-1)
    max_val = max_val.unsqueeze(-1).unsqueeze(-1)
    normalized[mask] = 255.0 * (tensor[mask] - min_val[mask]) / (max_val[mask] - min_val[mask])
    return normalized

def _fspecial_gaussian_torch(win_size, sigma):
    """Create a two-dimensional Gaussian filter similar to MATLAB's fspecial('gaussian') function using PyTorch."""
    coords = torch.arange(-win_size // 2 + 1, win_size // 2 + 1, dtype=torch.float64)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g /= g.sum()
    kernel = g.reshape(1, 1, win_size, win_size).contiguous()
    return kernel

def rgb2gray(tensor):
    """Convert RGB (B,3,H,W) to grayscale (B,1,H,W); return directly if already grayscale."""
    if tensor.shape[1] == 1:
        return tensor
    elif tensor.shape[1] == 3:
        # Standard BT.601 conversion coefficients
        r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
        gray = torch.round(0.299 * r + 0.587 * g + 0.114 * b)
        return gray.to(torch.float64)
    else:
        raise ValueError(f"channel must be 1 or 3, got {tensor.shape[1]}")
    
def compute_metrics(metric_funcs, fusion_pred, I1, I2):
    """Compute metrics for a set of metric functions"""
    results = {}
    for met_func in metric_funcs:
        _metric_name = met_func.__name__
        param_count = len(inspect.signature(met_func).parameters)

        if _metric_name in ["Metric_BiSWE", "Metric_MS2R"]:
            fusion_clip = torch.round(fusion_pred * 255)
            ir_clip = torch.round(I1[:, 1:4, :, :, :] * 255)
            rgb_clip = torch.round(I2[:, 1:4, :, :, :] * 255)
        elif _metric_name in ["Metric_VIF", "Metric_SSIM", "Metric_MI", "Metric_Qabf", "Metric_MEF_SSIM"]:
            fusion_clip = torch.round(fusion_pred[:, 1, :, :, :] * 255)
            ir_clip = torch.round(I1[:, 2, :, :, :] * 255)
            rgb_clip = torch.round(I2[:, 2, :, :, :] * 255)
        else:
            raise ValueError(f"Unsupported metric function: {_metric_name}")

        if param_count == 1:
            _metric = met_func(fusion_clip).item()
        elif param_count == 3:
            _metric = met_func(fusion_clip, ir_clip, rgb_clip).item()
        else:
            raise ValueError(
                f"Metric function {_metric_name} has unsupported number of parameters: {param_count}"
            )
        results[_metric_name] = _metric
    return results

def _ignore_nan(tensor):
    """Ignore NaNs in tensor and compute mean over valid channels"""
    not_nan_channel = ~torch.isnan(tensor)  # B,C
    tensor_without_nan = torch.nan_to_num(tensor, nan=0.0)
    return torch.sum(tensor_without_nan, dim=1).unsqueeze(1) / torch.sum(not_nan_channel, dim=1)

# Bi-Directional Self-Warping Error (BiSWE)
class BiSWE_Evaluator:
    def __init__(
        self,
        use_occlusion=True,
        occ_threshold=1.0,
        device="cuda",
        raft_config_path="config/module/spring-S.json",
    ):
        """
        flow_net: An optical flow model with a forward interface as flow_net(img1, img2)['final']
        use_occlusion: Whether to enable forward-backward consistency check
        occ_threshold: Occlusion detection threshold
        """
        from src.model.raft import RAFT
        from src.model.RAFT_component.raft_utils import load_ckpt
        from src.model.utils import load_args_from_json, flow_warp
        
        # Load optical flow
        self.raft_args = load_args_from_json(raft_config_path)
        self.flow_net = RAFT(self.raft_args).to(device).eval()
        load_ckpt(self.flow_net, self.raft_args.path)
        self.use_occlusion = use_occlusion
        self.occ_threshold = occ_threshold
        self.flow_warp = flow_warp

    @torch.no_grad()
    def occlusion_mask(self, img1, img2, flow_ab):  # img1 current frame, flow_ab is flow from img1 to img2
        """Compute forward-backward consistency occlusion mask."""
        flow_ba = self.flow_net(img2, img1)["final"]  # flow from img2 to img1, img2 is the reference frame
        flow_ba_warped = self.flow_warp(flow_ba, flow_ab)
        fb_diff = flow_ab + flow_ba_warped
        fb_consistency = fb_diff.norm(p=2, dim=1)  # [B, H, W]
        mask = (fb_consistency < self.occ_threshold).float() # mask at the current scale
        return mask

    @torch.no_grad()
    def evaluate(self, video_clip, R1_clip, R2_clip):
        """
        Note that the input to the optical flow network must be images in the range [0, 255]

        Returns:
            avg_error: float
            all_errors: [B] tensor
        """
        B, _, _, H, W = video_clip.shape
        device = video_clip.device

        cur = video_clip[:, 1]  # V_t
        prev = video_clip[:, 0]  # V_{t-1}
        nxt = video_clip[:, 2]  # V_{t+1}
        # print(self.use_occlusion)
        #R1_mask
        if self.use_occlusion:
            flow_R1_cur2prev = self.flow_net(R1_clip[:,1], R1_clip[:,0])["final"]
            flow_R1_cur2next = self.flow_net(R1_clip[:,1], R1_clip[:,2])["final"]

            flow_R2_cur2prev = self.flow_net(R2_clip[:,1], R2_clip[:,0])["final"]
            flow_R2_cur2next = self.flow_net(R2_clip[:,1], R2_clip[:,2])["final"]

            mask_R1_prev = self.occlusion_mask(R1_clip[:,1], R1_clip[:,0], flow_R1_cur2prev)
            mask_R1_next = self.occlusion_mask(R1_clip[:,1], R1_clip[:,2], flow_R1_cur2next)

            mask_R2_prev = self.occlusion_mask(R2_clip[:,1], R2_clip[:,0], flow_R2_cur2prev)
            mask_R2_next = self.occlusion_mask(R2_clip[:,1], R2_clip[:,2], flow_R2_cur2next)

            mask_prev = mask_R1_prev*mask_R2_prev
            mask_next = mask_R1_next*mask_R2_next
        else:
            mask_prev = torch.ones((B, H, W), device=device)
            mask_next = torch.ones((B, H, W), device=device)
            # print('nomask')

        flow_cur2prev = self.flow_net(cur,prev)["final"]
        flow_cur2next = self.flow_net(cur,nxt)["final"]        

        recon_prev = self.flow_warp(prev, flow_cur2prev)
        recon_next = self.flow_warp(nxt, flow_cur2next)

        diff_prev = (torch.abs(cur - recon_prev) ).mean(1)
        diff_next = (torch.abs(cur - recon_next) ).mean(1)

        err_prev = (mask_prev * diff_prev).sum(dim=(1, 2)) / (
            mask_prev.sum(dim=(1, 2)) + 1e-10
        )
        err_next = (mask_next * diff_next).sum(dim=(1, 2)) / (
            mask_next.sum(dim=(1, 2)) + 1e-10
        )

        total_error = err_prev + err_next  # [B]

        return total_error

# Motion Smoothness with Dual Reference Videos (MS2R)
class MS2R_Evaluator:
    def __init__(
        self,
        device="cuda",
        raft_config_path="config/module/spring-S.json",
        bin_range=(0.0, 10.0),
        bin_width=1.0,
    ):
        """
        device: Default is GPU
        raft_config_path: Optical flow model configuration path
        bin_range: L2 norm histogram range (min, max)
        bin_width: Histogram bin width
        """
        from src.model.raft import RAFT
        from src.model.RAFT_component.raft_utils import load_ckpt
        from src.model.utils import load_args_from_json

        self.device = device
        self.bin_range = bin_range
        self.bin_width = bin_width

        raft_args = load_args_from_json(raft_config_path)
        self.flow_net = RAFT(raft_args).to(device).eval()
        load_ckpt(self.flow_net, raft_args.path)

    @torch.no_grad()
    def compute_flow(self, img1, img2):
        return self.flow_net(img1, img2)["final"]

    @torch.no_grad()
    def compute_differential_flow(self, G0, G1, G2, R0, R1, R2):
        """Compute Differential Flow"""
        d_gen = self.compute_flow(G1, G2) - self.compute_flow(G0, G1)
        d_ref = self.compute_flow(R1, R2) - self.compute_flow(R0, R1)
        return d_gen - d_ref  # [1, 2, H, W]

    def compute_sample_smoothness(self, d_merged):
        """
        d_merged: [2, 2, H, W], differential flow after merging two reference videos
        Returns: a float value representing the smoothness metric for this sample
        """
        D_l2 = torch.norm(d_merged, dim=1).view(-1)  # [N]=[2 * H * W]

        # histogram calculation
        hist = torch.histc(
            D_l2,
            bins=int((self.bin_range[1] - self.bin_range[0]) / self.bin_width),
            min=self.bin_range[0],
            max=self.bin_range[1],
        )
        total = hist.sum() + 1e-10
        metric = (torch.log(hist + 1e-10) - torch.log(total)).sum()  # scalar
        return metric

    def compute_ms2r_metric(self, G_clip, R1_clip, R2_clip):
        """
        Outputs one smoothness value per sample

        G_clip:  [B, 3, C, H, W] -> Generated
        R1_clip: [B, 3, C, H, W] -> Reference 1
        R2_clip: [B, 3, C, H, W] -> Reference 2

        Returns:
            Tensor of shape [B]
        """
        B = G_clip.shape[0]
        all_metrics = []

        for k in range(B):
            G0, G1, G2 = G_clip[k]
            R10, R11, R12 = R1_clip[k]
            R20, R21, R22 = R2_clip[k]

            D1 = self.compute_differential_flow(
                G0.unsqueeze(0),
                G1.unsqueeze(0),
                G2.unsqueeze(0),
                R10.unsqueeze(0),
                R11.unsqueeze(0),
                R12.unsqueeze(0),
            )
            D2 = self.compute_differential_flow(
                G0.unsqueeze(0),
                G1.unsqueeze(0),
                G2.unsqueeze(0),
                R20.unsqueeze(0),
                R21.unsqueeze(0),
                R22.unsqueeze(0),
            )

            metric_val=0.5*(torch.mean(torch.abs(D1))+torch.mean(torch.abs(D2)))
            all_metrics.append(metric_val)

        return torch.stack(all_metrics)  # [B]



def Metric_BiSWE(fusion_clip: torch.Tensor, source_clip_1: torch.Tensor, source_clip_2: torch.Tensor):
    evaluator = BiSWE_Evaluator(use_occlusion=True)
    return evaluator.evaluate(fusion_clip, source_clip_1, source_clip_2)

def Metric_MS2R(fusion_clip: torch.Tensor, source_clip_1: torch.Tensor, source_clip_2: torch.Tensor):
    evaluator = MS2R_Evaluator()
    return evaluator.compute_ms2r_metric(fusion_clip, source_clip_1, source_clip_2)