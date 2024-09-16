import torch

BASE_SEQLEN = 16384
B_LIMITS = {
    1: 32,
    2: 16,
    4: 4,
    8: 4,
    16: 4,
    32: 4,
    64: 4,
    128: 4, 
    256: 4
}

H_LIMITS = {
    8: 384,
    16: 192,
    32: 96,
    64: 48,
    128: 32,
    256: 16
}

def set_B_H(B, H, seqlen):
    seqlen_factor = seqlen // BASE_SEQLEN

    if seqlen_factor in B_LIMITS and B > B_LIMITS[seqlen_factor]:
        B = B_LIMITS[seqlen_factor]

    if seqlen_factor in H_LIMITS and H > H_LIMITS[seqlen_factor]:
        H = H_LIMITS[seqlen_factor]

    return B, H

def ref_fft_conv(u, k, n=None):
    if n is None:
        n = u.size(-1)
    l = u.size(-1)
    u_f = torch.fft.fft(u.to(torch.float32), n=n)
    k_f = torch.fft.fft(k.to(torch.float32), n=n)
    u_f = u_f * k_f
    out = torch.fft.ifft(u_f, n=n)
    return out.real.to(u.dtype)[..., :l]