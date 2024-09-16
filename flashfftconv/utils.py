import torch

def fft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(-2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_fft(n, m):
    """Compute the twiddle factors of size n x m"""
    # n_a = torch.arange(n).view(-1, 1)
    # m_a = torch.arange(m)
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(-2j * torch.pi * n_a * m_a / N)
    return M

def ifft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_ifft(n, m):
    """Compute the twiddle factors of size n x m"""
    # n_a = torch.arange(n).view(-1, 1)
    # m_a = torch.arange(m)
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(2j * torch.pi * n_a * m_a / N)
    return M

def monarch_outer_dft(x, f_sqrt_N_fft, twiddle_factors_fft, sqrt_N):
    x = x.transpose(-1, -2) # 32K, 32
    x = x @ f_sqrt_N_fft    # 32K, 32
    x = x.transpose(-1, -2) # 32, 32K
    # x = (f_sqrt_N_fft.T @ x) * twiddle_factors_fft # (32, 32K) * (32, 32K), pointwise

    return (x * twiddle_factors_fft).contiguous()

def monarch_outer_idft(x, f_sqrt_N_ifft, twiddle_factors_ifft, sqrt_N):
    # x = f_sqrt_N_ifft.T @ (x * twiddle_factors_ifft) # (32, 32K) * (32, 32K), pointwise
    x = x * twiddle_factors_ifft 
    x = x.transpose(-1, -2) # 32K, 32
    x = x @ f_sqrt_N_ifft
    x = x.transpose(-1, -2) # 32, 32K

    return x.contiguous()