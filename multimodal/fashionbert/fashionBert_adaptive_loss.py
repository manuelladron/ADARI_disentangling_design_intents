
import torch


def adaptive_loss(outputs):
    masked_lm_loss = outputs['masked_lm_loss'] 
    masked_patch_loss = outputs['masked_patch_loss'] 
    alignment_loss = outputs['alignment_loss']
    
    G = torch.stack([masked_lm_loss, alignment_loss, masked_patch_loss]) # [3]
    w0 = 1.0
    w1 = 1.0
    w2 = 1.0
    isAdaptive = True
    if isAdaptive:
        nG = torch.sqrt(torch.nn.LogSoftmax(dim=0)(G))
        alpha = 1.0
        K = 3.0
        denominator = (alpha * K - nG[0]) * (alpha * K - nG[1]) + (alpha * K - nG[1]) * (alpha * K - nG[2]) +                     (alpha * K - nG[2]) * (alpha * K - nG[0])
        w0 = (alpha * K - nG[1]) * (alpha * K - nG[2]) / denominator
        w1 = (alpha * K - nG[2]) * (alpha * K - nG[0]) / denominator
        w2 = (alpha * K - nG[0]) * (alpha * K - nG[1]) / denominator

    adaptive_loss = w0 * masked_lm_loss + w1 * alignment_loss + w2 * masked_patch_loss
    return adaptive_loss
    






