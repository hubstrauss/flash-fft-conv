import torch
from flashfftconv import FlashFFTConv
from utils import *
import pytest

@pytest.mark.parametrize('B', [1, 2, 4, 8, 64])
@pytest.mark.parametrize('H', [768, 111])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [256, 512, 1024, 4096, 8192, 16384, 32768, 2 * 32768, 4 * 32768, 8 * 32768, 16 * 32768, 32 * 32768, 64 * 32768, 128 * 32768])
def test_flash_fft_conv(B, H, seqlen, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(0)
    device = 'cuda'
    B, H = set_B_H(B, H, seqlen)

    N = seqlen
    u = torch.randn(B, H, N, device=device).to(dtype) * 0.02
    u[:, :, seqlen // 2 :] = 0.
    k = torch.randn(H, N, device=device) * 0.02
    k[:, seqlen // 2 :] = 0.
    mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen]
    k = k * mask

    u_clone = u.clone()
    k_clone = k.clone()

    u.requires_grad = True
    k.requires_grad = True
    u_clone.requires_grad = True
    k_clone.requires_grad = True

    fftconv = FlashFFTConv(seqlen, dtype=dtype).to(device)

    fftconv.requires_grad = True
    
    ref_out = ref_fft_conv(u_clone, k_clone, n = seqlen)
    out = fftconv(u, k)

    if not torch.allclose(out, ref_out, atol=1e-2):
        breakpoint()
    assert(torch.allclose(out, ref_out, atol=1e-2))

    dout = torch.randn_like(out) * 0.02
    dout_clone = dout.clone()

    # initialize the gradients
    # see https://github.com/pytorch/pytorch/issues/109448
    u.backward(dout, retain_graph=True)
    k.backward(dout[0], retain_graph=True)
    u.grad.data.zero_()
    k.grad.data.zero_()

    u_clone.backward(dout, retain_graph=True)
    k_clone.backward(dout[0], retain_graph=True)
    u_clone.grad.data.zero_()
    k_clone.grad.data.zero_()

    ref_out.backward(dout_clone, retain_graph=True)
    out.backward(dout, retain_graph=True)

    assert(torch.allclose(u.grad, u_clone.grad, atol=1e-2))

    ktol = 1e-1 if seqlen < 16 * 32768 else 1 if seqlen < 128 * 32768 else 2

    assert(torch.allclose(k.grad, k_clone.grad, atol=ktol)) # larger error for k.grad since no scaling

@pytest.mark.parametrize('B', [1, 2, 4, 8, 64])
@pytest.mark.parametrize('H', [768, 111])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [256, 512, 1024, 4096, 8192, 16384, 32768, 2 * 32768, 4 * 32768, 8 * 32768, 16 * 32768, 32 * 32768, 64 * 32768, 128 * 32768])
def test_flash_fft_conv_padded(B, H, seqlen, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(0)
    device = 'cuda'
    B, H = set_B_H(B, H, seqlen)

    N = seqlen
    u = torch.randn(B, H, N // 2, device=device).to(dtype) * 0.02
    k = torch.randn(H, N // 2, device=device) * 0.02
    mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen // 2]
    k = k * mask

    u_clone = u.clone()
    k_clone = k.clone()

    u.requires_grad = True
    k.requires_grad = True
    u_clone.requires_grad = True
    k_clone.requires_grad = True

    fftconv = FlashFFTConv(seqlen, dtype=dtype).to(device)

    fftconv.requires_grad = True
    
    ref_out = ref_fft_conv(u_clone, k_clone, n = seqlen)
    out = fftconv(u, k)

    if not torch.allclose(out, ref_out, atol=1e-2):
        breakpoint()
    assert(torch.allclose(out, ref_out, atol=1e-2))

    dout = torch.randn_like(out) * 0.02
    dout_clone = dout.clone()

    # initialize the gradients
    # see https://github.com/pytorch/pytorch/issues/109448
    u.backward(dout, retain_graph=True)
    k.backward(dout[0], retain_graph=True)
    u.grad.data.zero_()
    k.grad.data.zero_()

    u_clone.backward(dout, retain_graph=True)
    k_clone.backward(dout[0], retain_graph=True)
    u_clone.grad.data.zero_()
    k_clone.grad.data.zero_()

    ref_out.backward(dout_clone, retain_graph=True)
    out.backward(dout, retain_graph=True)

    assert(torch.allclose(u.grad, u_clone.grad, atol=1e-2))

    ktol = 1e-1 if seqlen < 16 * 32768 else 1 if seqlen < 128 * 32768 else 2

    assert(torch.allclose(k.grad, k_clone.grad, atol=ktol)) # larger error for k.grad since no scaling

@pytest.mark.parametrize('B', [1, 2, 4, 8, 64])
@pytest.mark.parametrize('H', [768, 111])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [256, 512, 1024, 4096, 8192, 16384, 32768, 2 * 32768, 4 * 32768, 8 * 32768, 16 * 32768, 32 * 32768, 64 * 32768, 128 * 32768])
def test_flash_fft_conv_gating(B, H, seqlen, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(0)
    device = 'cuda'
    B, H = set_B_H(B, H, seqlen)

    N = seqlen
    u = torch.randn(B, H, N, device=device).to(dtype) * 0.02
    u[:, :, seqlen // 2 :] = 0.
    pregate = torch.randn(B, H, N, device=device).to(dtype) * 0.02
    pregate[:, :, seqlen // 2 :] = 0.
    postgate = torch.randn(B, H, N, device=device).to(dtype) * 0.02
    postgate[:, :, seqlen // 2 :] = 0.
    k = torch.randn(H, N, device=device) * 0.02
    k[:, seqlen // 2 :] = 0.
    mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen]
    k = k * mask

    u_clone = u.clone()
    k_clone = k.clone()
    pregate_clone = pregate.clone()
    postgate_clone = postgate.clone()

    u.requires_grad = True
    k.requires_grad = True
    pregate.requires_grad = True
    postgate.requires_grad = True
    u_clone.requires_grad = True
    k_clone.requires_grad = True
    pregate_clone.requires_grad = True
    postgate_clone.requires_grad = True

    fftconv = FlashFFTConv(seqlen, dtype=dtype).to(device)

    fftconv.requires_grad = True
    
    ref_out = ref_fft_conv(u_clone * pregate_clone, k_clone, n = seqlen) * postgate_clone
    out = fftconv(u, k, pregate, postgate)

    if not torch.allclose(out, ref_out, atol=1e-2):
        breakpoint()
    assert(torch.allclose(out, ref_out, atol=1e-2))

    dout = torch.randn_like(out) * 0.02
    dout_clone = dout.clone()

    # initialize the gradients
    # see https://github.com/pytorch/pytorch/issues/109448
    u.backward(dout, retain_graph=True)
    k.backward(dout[0], retain_graph=True)
    pregate.backward(dout, retain_graph=True)
    postgate.backward(dout, retain_graph=True)
    u.grad.data.zero_()
    k.grad.data.zero_()
    pregate.grad.data.zero_()
    postgate.grad.data.zero_()

    u_clone.backward(dout, retain_graph=True)
    k_clone.backward(dout[0], retain_graph=True)
    pregate_clone.backward(dout, retain_graph=True)
    postgate_clone.backward(dout, retain_graph=True)
    u_clone.grad.data.zero_()
    k_clone.grad.data.zero_()
    pregate_clone.grad.data.zero_()
    postgate_clone.grad.data.zero_()

    ref_out.backward(dout_clone, retain_graph=True)
    out.backward(dout, retain_graph=True)

    assert(torch.allclose(u.grad, u_clone.grad, atol=1e-2))
    assert(torch.allclose(pregate.grad, pregate_clone.grad, atol=1e-2))
    assert(torch.allclose(postgate.grad, postgate_clone.grad, atol=1e-2))

    ktol = 1e-1 if seqlen < 16 * 32768 else 1 if seqlen < 128 * 32768 else 2

    assert(torch.allclose(k.grad, k_clone.grad, atol=ktol)) # larger error for k.grad since no scaling

@pytest.mark.parametrize('B', [1, 2, 4, 8, 64])
@pytest.mark.parametrize('H', [768, 111])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [256, 512, 1024, 4096, 8192, 16384, 32768, 2 * 32768, 4 * 32768, 8 * 32768, 16 * 32768, 32 * 32768, 64 * 32768, 128 * 32768])
def test_flash_fft_conv_gating_padded(B, H, seqlen, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(0)
    device = 'cuda'
    B, H = set_B_H(B, H, seqlen)

    N = seqlen
    u = torch.randn(B, H, N // 2, device=device).to(dtype) * 0.02
    pregate = torch.randn(B, H, N // 2, device=device).to(dtype) * 0.02
    postgate = torch.randn(B, H, N // 2, device=device).to(dtype) * 0.02
    k = torch.randn(H, N // 2, device=device) * 0.02
    mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen // 2]
    k = k * mask

    u_clone = u.clone()
    k_clone = k.clone()
    pregate_clone = pregate.clone()
    postgate_clone = postgate.clone()

    u.requires_grad = True
    k.requires_grad = True
    pregate.requires_grad = True
    postgate.requires_grad = True
    u_clone.requires_grad = True
    k_clone.requires_grad = True
    pregate_clone.requires_grad = True
    postgate_clone.requires_grad = True

    fftconv = FlashFFTConv(seqlen, dtype=dtype).to(device)

    fftconv.requires_grad = True
    
    ref_out = ref_fft_conv(u_clone * pregate_clone, k_clone, n = seqlen) * postgate_clone
    out = fftconv(u, k, pregate, postgate)

    if not torch.allclose(out, ref_out, atol=1e-2):
        breakpoint()
    assert(torch.allclose(out, ref_out, atol=1e-2))

    dout = torch.randn_like(out) * 0.02
    dout_clone = dout.clone()

    # initialize the gradients
    # see https://github.com/pytorch/pytorch/issues/109448
    u.backward(dout, retain_graph=True)
    k.backward(dout[0], retain_graph=True)
    pregate.backward(dout, retain_graph=True)
    postgate.backward(dout, retain_graph=True)
    u.grad.data.zero_()
    k.grad.data.zero_()
    pregate.grad.data.zero_()
    postgate.grad.data.zero_()

    u_clone.backward(dout, retain_graph=True)
    k_clone.backward(dout[0], retain_graph=True)
    pregate_clone.backward(dout, retain_graph=True)
    postgate_clone.backward(dout, retain_graph=True)
    u_clone.grad.data.zero_()
    k_clone.grad.data.zero_()
    pregate_clone.grad.data.zero_()
    postgate_clone.grad.data.zero_()

    ref_out.backward(dout_clone, retain_graph=True)
    out.backward(dout, retain_graph=True)

    assert(torch.allclose(u.grad, u_clone.grad, atol=1e-2))
    assert(torch.allclose(pregate.grad, pregate_clone.grad, atol=1e-2))
    assert(torch.allclose(postgate.grad, postgate_clone.grad, atol=1e-2))

    ktol = 1e-1 if seqlen < 16 * 32768 else 1 if seqlen < 128 * 32768 else 2

    assert(torch.allclose(k.grad, k_clone.grad, atol=ktol)) # larger error for k.grad since no scaling