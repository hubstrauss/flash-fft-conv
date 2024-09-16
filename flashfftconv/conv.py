# Copyright (c) 2023, Dan Fu and Hermann Kumbong.
import math

import torch
import torch.nn.functional as F

from einops import rearrange

from flashfftconv.utils import *
from flashfftconv.strategies import *
from monarch_cuda import monarch_conv_forward, monarch_conv_backward, \
    monarch_conv_forward_r2r, monarch_conv_backward_r2r, \
    monarch_conv_forward_16_16_16, monarch_conv_backward_16_16_16, \
    monarch_conv_forward_32_16_16, monarch_conv_backward_32_16_16, \
    monarch_conv_forward_16_32_32, monarch_conv_backward_16_32_32, \
    monarch_conv_forward_32_32_32, monarch_conv_backward_32_32_32, \
    monarch_conv_forward_16_16_16_complex, monarch_conv_backward_16_16_16_complex, \
    monarch_conv_forward_32_16_16_complex, monarch_conv_backward_32_16_16_complex, \
    monarch_conv_forward_16_32_32_complex, monarch_conv_backward_16_32_32_complex, \
    monarch_conv_forward_32_32_32_complex, monarch_conv_backward_32_32_32_complex
from monarch_cuda import butterfly_forward, butterfly_ifft_forward, butterfly_padded_forward, butterfly_ifft_padded_forward, butterfly_padded_gated_forward, butterfly_ifft_padded_gated_forward
from monarch_cuda import butterfly_bf16_forward, butterfly_ifft_bf16_forward, butterfly_padded_bf16_forward, butterfly_ifft_padded_bf16_forward, butterfly_padded_gated_bf16_forward, butterfly_ifft_padded_gated_bf16_forward

torch.manual_seed(23)

class FlashFFTConv(torch.nn.Module):
    def __init__(self, seqlen, dtype=torch.float16, use_32_butterfly=True):
        super().__init__()
        assert dtype == torch.bfloat16 or dtype == torch.float16
        self.seqlen = seqlen
        self.dtype = dtype
        self.use_32_butterfly=use_32_butterfly
        self._select_strategy()

    def _select_strategy(self):
        strategies = {
            (256, 1024): Strategy256_1024,
            (512, 2048): Strategy512_2048,
            (4096,): Strategy4096,
            #TODO: add the other configs
        }

        for seqlen_values, strategy_class in strategies.items():
            if self.seqlen in seqlen_values:
                strategy = strategy_class(self, self.dtype, self.use_32_butterfly)
                strategy.initialize()
                return

        raise NotImplementedError(f'seqlen {self.seqlen} not supported')
    
    def forward(self, u, k, pregate=None, postgate=None):
        if pregate is not None or postgate is not None:
            assert pregate is not None and postgate is not None
            return GatedFlashFFTConvFunc.apply(u, k, self, pregate, postgate)
        return FlashFFTConvFunc.apply(u, k, self)

class FlashFFTConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, fftconv_data):

        B, H, L = u.shape

        # replace this with a kernel
        if fftconv_data.seqlen in [512, 2048]:
            k_f = torch.fft.rfft(k, n=fftconv_data.seqlen)
        else:
            k_f = torch.fft.fft(k, n=fftconv_data.seqlen)

        ctx.fftconv_data = fftconv_data
        ctx.k_len = k.shape[-1]

        if fftconv_data.seqlen in [256, 1024]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted)

            return monarch_conv_forward(
                u, 
                k_f_permuted,
                fftconv_data.f_sqrt_N_fft, 
                fftconv_data.twiddle_factors_fft,
                fftconv_data.f_sqrt_N_ifft, 
                fftconv_data.twiddle_factors_ifft,
                None, 
                None,
                N, 
                L, 
                sqrt_N
            )
        elif fftconv_data.seqlen in [512, 2048]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N

            k_f = torch.view_as_real(k_f).to(fftconv_data.dtype).contiguous()
            
            if fftconv_data.training:
                ctx.save_for_backward(u, k_f)

            return monarch_conv_forward_r2r(
                u, k_f,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.twid,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                None, None,
                N, L, sqrt_N
            )
        elif fftconv_data.seqlen == 4096:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N
            sqrt_N_256 = fftconv_data.sqrt_N_256

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, sqrt_N_256, sqrt_N).transpose(-1, -2).reshape(H, sqrt_N, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted)

            out = monarch_conv_forward_16_16_16(
                u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft,
                fftconv_data.twiddle_factors_fft_16_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_sqrt_N_ifft,
                fftconv_data.twiddle_factors_ifft_16_256, fftconv_data.twiddle_factors_ifft_16_16,
                None, None,
                N, L, sqrt_N_256, sqrt_N
            )

            return out
        elif fftconv_data.seqlen == 8192:
            N = fftconv_data.N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, 256, 32).transpose(-1, -2).reshape(H, 32, 16, 16).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted)

            return monarch_conv_forward_32_16_16(
                u, k_f_permuted,
                fftconv_data.f_32_fft, fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_32_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_32_ifft, fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_32_256, fftconv_data.twiddle_factors_ifft_16_16,
                None, None,
                N, L
            )
        elif fftconv_data.seqlen == 16384:
            N = fftconv_data.N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, 1024, 16).transpose(-1, -2).reshape(H, 16, 32, 32).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted)

            return monarch_conv_forward_16_32_32(
                u, k_f_permuted,
                fftconv_data.f_16_fft, fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_16_1K, fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_16_ifft, fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_16_1K, fftconv_data.twiddle_factors_ifft_32_32,
                None, None,
                N, L
            )
        elif fftconv_data.seqlen == 32768:
            N = fftconv_data.N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, 1024, 32).transpose(-1, -2).reshape(H, 32, 32, 32).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted)
            
            return monarch_conv_forward_32_32_32(
                u, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                None, None,
                N, L
            )
        elif fftconv_data.seqlen == 16 * 4096:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 4096, 16).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 16, 256, 16).transpose(-1, -2).reshape(H, 16, 16, 16, 16).transpose(-1, -2).reshape(H * 16, 4096)).contiguous().to(fftconv_data.dtype)

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted)

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_16_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_16_fft_real,
                        fftconv_data.f_16_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096
                    )
            else:
                x = u.reshape(B, H, 16, 4096)
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_16_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                else:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_16_fft_real,
                        fftconv_data.f_16_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )

            x_half_real = x_half_real.reshape(B, H * 16, 4096)
            x_half_imag = x_half_imag.reshape(B, H * 16, 4096)

            out_half_real, out_half_imag = monarch_conv_forward_16_16_16_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            if L < N:
                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_16_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    x = butterfly_ifft_padded_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_16_ifft_real,
                        fftconv_data.f_16_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                out_half_real = out_half_real.reshape(B, H, 16, 4096)
                out_half_imag = out_half_imag.reshape(B, H, 16, 4096)

                if x.dtype == torch.float16:
                    out_half = butterfly_ifft_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_16_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                else:
                    out_half = butterfly_ifft_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_16_ifft_real,
                        fftconv_data.f_16_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                x = out_half.reshape(B, H, N)
            
            return x[..., :L]
        elif fftconv_data.seqlen == 16 * 8192:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:

                k_f_permuted = k_f.reshape(H, 4096, 32).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 256, 16).transpose(-1, -2).reshape(H, 32, 16, 16, 16).transpose(-1, -2).reshape(H * 32, 4096)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted)

                # assert(N == L)
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            4096
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            4096
                        )
                else:
                    x = u.reshape(B, H, 32, 4096)
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 32, 4096)
                x_half_imag = x_half_imag.reshape(B, H * 32, 4096)

                out_half_real, out_half_imag = monarch_conv_forward_16_16_16_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_16_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_16_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    4096, 4096
                )

                if L < N:
                    out_half_real = out_half_real.reshape(B, H, N)
                    out_half_imag = out_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        x = butterfly_ifft_padded_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        x = butterfly_ifft_padded_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    out_half_real = out_half_real.reshape(B, H, 32, 4096)
                    out_half_imag = out_half_imag.reshape(B, H, 32, 4096)

                    if x.dtype == torch.float16:
                        out_half = butterfly_ifft_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    else:
                        out_half = butterfly_ifft_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    x = out_half.reshape(B, H, N)
            else:

                k_f_permuted = k_f.reshape(H, 8192, 16).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 16, 256, 32).transpose(-1, -2).reshape(H, 16, 32, 16, 16).transpose(-1, -2).reshape(H * 16, 8192)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted)

                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                else:
                    x = u.reshape(B, H, 16, 8192)
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 16, 8192)
                x_half_imag = x_half_imag.reshape(B, H * 16, 8192)

                out_half_real, out_half_imag = monarch_conv_forward_32_16_16_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_32_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_32_ifft,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_32_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    8192, 8192
                )

                if L < N:
                    out_half_real = out_half_real.reshape(B, H, N)
                    out_half_imag = out_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        x = butterfly_ifft_padded_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        x = butterfly_ifft_padded_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    out_half_real = out_half_real.reshape(B, H, 16, 8192)
                    out_half_imag = out_half_imag.reshape(B, H, 16, 8192)
                    
                    if x.dtype == torch.float16:
                        out_half = butterfly_ifft_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    else:
                        out_half = butterfly_ifft_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    x = out_half.reshape(B, H, N)

            return x[..., :L]
        elif fftconv_data.seqlen == 16 * 16384:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:

                k_f_permuted = k_f.reshape(H, 8192, 32).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 256, 32).transpose(-1, -2).reshape(H, 32, 32, 16, 16).transpose(-1, -2).reshape(H * 32, 8192)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted)

                # assert(N == L)
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                else:
                    x = u.reshape(B, H, 32, 8192)
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 32, 8192)
                x_half_imag = x_half_imag.reshape(B, H * 32, 8192)

                out_half_real, out_half_imag = monarch_conv_forward_32_16_16_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_32_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_32_ifft,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_32_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    8192, 8192
                )

                if L < N:
                    out_half_real = out_half_real.reshape(B, H, N)
                    out_half_imag = out_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        x = butterfly_ifft_padded_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        x = butterfly_ifft_padded_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    out_half_real = out_half_real.reshape(B, H, 32, 8192)
                    out_half_imag = out_half_imag.reshape(B, H, 32, 8192)

                    if x.dtype == torch.float16:
                        out_half = butterfly_ifft_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    else:
                        out_half = butterfly_ifft_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    x = out_half.reshape(B, H, N)
            else:

                k_f_permuted = k_f.reshape(H, 16384, 16).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 16, 1024, 16).transpose(-1, -2).reshape(H, 16, 16, 32, 32).transpose(-1, -2).reshape(H * 16, 16384)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted)

                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                else:
                    x = u.reshape(B, H, 16, 16384)
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 16, 16384)
                x_half_imag = x_half_imag.reshape(B, H * 16, 16384)

                out_half_real, out_half_imag = monarch_conv_forward_16_32_32_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_16_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_16_ifft,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_16_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    16384, 16384
                )

                if L < N:
                    out_half_real = out_half_real.reshape(B, H, N)
                    out_half_imag = out_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        x = butterfly_ifft_padded_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        x = butterfly_ifft_padded_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    out_half_real = out_half_real.reshape(B, H, 16, 16384)
                    out_half_imag = out_half_imag.reshape(B, H, 16, 16384)

                    if x.dtype == torch.float16:
                        out_half = butterfly_ifft_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    else:
                        out_half = butterfly_ifft_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    x = out_half.reshape(B, H, N)

            return x[..., :L]
        elif fftconv_data.seqlen == 16 * 32768:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:
                k_f_permuted = k_f.reshape(H, 16384, 32).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 1024, 16).transpose(-1, -2).reshape(H, 32, 16, 32, 32).transpose(-1, -2).reshape(H * 32, 16384)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted)

                # assert(N == L)
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                else:
                    x = u.reshape(B, H, 32, 16384)
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 32, 16384)
                x_half_imag = x_half_imag.reshape(B, H * 32, 16384)

                out_half_real, out_half_imag = monarch_conv_forward_16_32_32_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_16_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_16_ifft,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_16_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    16384, 16384
                )

                if L < N:
                    out_half_real = out_half_real.reshape(B, H, N)
                    out_half_imag = out_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        x = butterfly_ifft_padded_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        x = butterfly_ifft_padded_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    out_half_real = out_half_real.reshape(B, H, 32, 16384)
                    out_half_imag = out_half_imag.reshape(B, H, 32, 16384)

                    if x.dtype == torch.float16:
                        out_half = butterfly_ifft_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    else:
                        out_half = butterfly_ifft_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    x = out_half.reshape(B, H, N)
            else:
                k_f_permuted = k_f.reshape(H, 32768, 16).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 16, 1024, 32).transpose(-1, -2).reshape(H, 16, 32, 32, 32).transpose(-1, -2).reshape(H * 16, 32768)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted)

                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            32768
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            32768
                        )
                else:
                    x = u.reshape(B, H, 16, 32768)
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 16, 32768)
                x_half_imag = x_half_imag.reshape(B, H * 16, 32768)

                out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_32_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_32_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    32768, 32768
                )

                if L < N:
                    out_half_real = out_half_real.reshape(B, H, N)
                    out_half_imag = out_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        x = butterfly_ifft_padded_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        x = butterfly_ifft_padded_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    out_half_real = out_half_real.reshape(B, H, 16, 32768)
                    out_half_imag = out_half_imag.reshape(B, H, 16, 32768)
                    
                    if x.dtype == torch.float16:
                        out_half = butterfly_ifft_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    else:
                        out_half = butterfly_ifft_bf16_forward(
                            out_half_real, out_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    x = out_half.reshape(B, H, N)

            return x[..., :L]
        elif fftconv_data.seqlen == 32 * 32768:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 32768, 32).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 1024, 32).transpose(-1, -2).reshape(H, 32, 32, 32, 32).transpose(-1, -2).reshape(H * 32, 32768)).contiguous().to(fftconv_data.dtype)
            
            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted)

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
            else:
                x = u.reshape(B, H, 32, 32768)
                
                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                else:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                
            x_half_real = x_half_real.reshape(B, H * 32, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 32, 32768)

            out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            if L < N:
                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)
                
                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    x = butterfly_ifft_padded_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                out_half_real = out_half_real.reshape(B, H, 32, 32768)
                out_half_imag = out_half_imag.reshape(B, H, 32, 32768)

                if x.dtype == torch.float16:
                    out_half = butterfly_ifft_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                else:
                    out_half = butterfly_ifft_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                x = out_half.reshape(B, H, N)

            return x[..., :L]
        elif fftconv_data.seqlen == 64 * 32768:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 32768, 64).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 64, 1024, 32).transpose(-1, -2).reshape(H, 64, 32, 32, 32).transpose(-1, -2).reshape(H * 64, 32768)).contiguous().to(fftconv_data.dtype)

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted)

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_64_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_64_fft_real,
                        fftconv_data.f_64_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
            else:
                x = u.reshape(B, H, 64, 32768)
                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_64_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                else:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_64_fft_real,
                        fftconv_data.f_64_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )

            x_half_real = x_half_real.reshape(B, H * 64, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 64, 32768)

            out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            if L < N:
                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_64_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    x = butterfly_ifft_padded_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_64_ifft_real,
                        fftconv_data.f_64_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                out_half_real = out_half_real.reshape(B, H, 64, 32768)
                out_half_imag = out_half_imag.reshape(B, H, 64, 32768)

                if x.dtype == torch.float16:
                    out_half = butterfly_ifft_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_64_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                else:
                    out_half = butterfly_ifft_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_64_ifft_real,
                        fftconv_data.f_64_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                x = out_half.reshape(B, H, N)

            return x[..., :L]    
        elif fftconv_data.seqlen == 128 * 32768:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 32768, 128).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 128, 1024, 32).transpose(-1, -2).reshape(H, 128, 32, 32, 32).transpose(-1, -2).reshape(H * 128, 32768)).contiguous().to(fftconv_data.dtype)

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted)

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_128_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_128_fft_real,
                        fftconv_data.f_128_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
            else:
                x = u.reshape(B, H, 128, 32768)
                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_128_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                else:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_128_fft_real,
                        fftconv_data.f_128_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )

            x_half_real = x_half_real.reshape(B, H * 128, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 128, 32768)

            out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            if L < N:
                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_128_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    x = butterfly_ifft_padded_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_128_ifft_real,
                        fftconv_data.f_128_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                out_half_real = out_half_real.reshape(B, H, 128, 32768)
                out_half_imag = out_half_imag.reshape(B, H, 128, 32768)
                
                if x.dtype == torch.float16:
                    out_half = butterfly_ifft_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_128_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                else:
                    out_half = butterfly_ifft_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_128_ifft_real,
                        fftconv_data.f_128_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                x = out_half.reshape(B, H, N)

            return x[..., :L]
        else:
            raise NotImplementedError(f'seqlen {fftconv_data.seqlen} not supported for FlashFFTConv fwd')

    @staticmethod
    def backward(ctx, dout):
        fftconv_data = ctx.fftconv_data
        # assert(dout.dtype == fftconv_data.dtype)

        B, H, L = dout.shape
        dout = dout.contiguous()

        u, k_f_permuted = ctx.saved_tensors
        k_len = ctx.k_len

        if fftconv_data.seqlen in [256, 1024]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N

            du, dk_f_permuted = monarch_conv_backward(
                dout, u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                None, None,
                N, L, sqrt_N
            )
            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None
        elif fftconv_data.seqlen in [512, 2048]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N
            
            du, dk_f = monarch_conv_backward_r2r(
                dout, u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.twid,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                None, None,
                N, L, sqrt_N
            )
            dk_f = torch.fft.irfft(
                torch.view_as_complex(dk_f.to(torch.float32)), n=fftconv_data.seqlen, norm='forward'
            ).real[..., :k_len] / 2

            return du, dk_f, None
        elif fftconv_data.seqlen == 4096:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N
            sqrt_N_256 = fftconv_data.sqrt_N_256

            du, dk_f_permuted = monarch_conv_backward_16_16_16(
                dout, u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft,
                fftconv_data.twiddle_factors_fft_16_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_sqrt_N_ifft,
                fftconv_data.twiddle_factors_ifft_16_256, fftconv_data.twiddle_factors_ifft_16_16,
                None, None,
                N, L, sqrt_N_256, sqrt_N
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, sqrt_N, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, sqrt_N, sqrt_N_256).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None
        elif fftconv_data.seqlen == 8192:
            N = fftconv_data.N

            # assert(L == N)

            du, dk_f_permuted = monarch_conv_backward_32_16_16(
                dout, u, k_f_permuted,
                fftconv_data.f_32_fft, fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_32_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_32_ifft, fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_32_256, fftconv_data.twiddle_factors_ifft_16_16,
                None, None,
                N, L
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 16, 16).transpose(-1, -2).reshape(H, 32, 256).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None
        elif fftconv_data.seqlen == 16384:
            N = fftconv_data.N

            # assert(L == N)

            du, dk_f_permuted = monarch_conv_backward_16_32_32(
                dout, u, k_f_permuted,
                fftconv_data.f_16_fft, fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_16_1K, fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_16_ifft, fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_16_1K, fftconv_data.twiddle_factors_ifft_32_32,
                None, None,
                N, L
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 32, 32).transpose(-1, -2).reshape(H, 16, 1024).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None
        elif fftconv_data.seqlen == 32768:
            N = fftconv_data.N

            # assert(L == N)

            du, dk_f_permuted = monarch_conv_backward_32_32_32(
                dout, u, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                None, None,
                N, L
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 32, 32).transpose(-1, -2).reshape(H, 32, 1024).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None
        elif fftconv_data.seqlen == 16 * 4096:
            N = fftconv_data.N

            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_16_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_forward(
                        dout,
                        fftconv_data.f_16_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_16_fft_real,
                        fftconv_data.f_16_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                        dout,
                        fftconv_data.f_16_fft_real,
                        fftconv_data.f_16_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096
                    )
            else:
                x = u.reshape(B, H, 16, 4096)
                dout = dout.reshape(B, H, 16, 4096)

                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_16_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_forward(
                        dout,
                        fftconv_data.f_16_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_16_fft_real,
                        fftconv_data.f_16_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_bf16_forward(
                        dout,
                        fftconv_data.f_16_fft_real,
                        fftconv_data.f_16_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )

            x_half_real = x_half_real.reshape(B, H * 16, 4096)
            x_half_imag = x_half_imag.reshape(B, H * 16, 4096)

            dout_half_real = dout_half_real.reshape(B, H * 16, 4096)
            dout_half_imag = dout_half_imag.reshape(B, H * 16, 4096)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_16_16_complex(
                dout_half_real, dout_half_imag,
                x_half_real, x_half_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            if L < N:
                dx_half_real = dx_half_real.reshape(B, H, N)
                dx_half_imag = dx_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    dx = butterfly_ifft_padded_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_16_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    dx = butterfly_ifft_padded_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_16_ifft_real,
                        fftconv_data.f_16_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                dx_half_real = dx_half_real.reshape(B, H, 16, 4096)
                dx_half_imag = dx_half_imag.reshape(B, H, 16, 4096)

                if x.dtype == torch.float16:
                    dx_half = butterfly_ifft_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_16_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    dx_half = butterfly_ifft_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_16_ifft_real,
                        fftconv_data.f_16_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                dx = dx_half.reshape(B, H, N)

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 16, 16, 16).transpose(-1, -2).reshape(H, 16, 16, 256).transpose(-1, -2).reshape(H, 16, 4096).transpose(-1, -2).reshape(H, N) * 16,
                norm='forward', n=N
            ).real[..., :k_len]

            return dx[..., :L], dk_f, None           
        elif fftconv_data.seqlen == 16 * 8192:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            4096
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_forward(
                            dout,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            4096
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            4096
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                            dout,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            4096
                        )
                else:
                    x = u.reshape(B, H, 32, 4096)
                    dout = dout.reshape(B, H, 32, 4096)

                    
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_forward(
                            dout,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_bf16_forward(
                            dout,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                
                x_half_real = x_half_real.reshape(B, H * 32, 4096)
                x_half_imag = x_half_imag.reshape(B, H * 32, 4096)

                dout_half_real = dout_half_real.reshape(B, H * 32, 4096)
                dout_half_imag = dout_half_imag.reshape(B, H * 32, 4096)

                dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_16_16_complex(
                    dout_half_real, dout_half_imag,
                    x_half_real, x_half_imag, k_f_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_16_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_16_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    4096, 4096
                )

                if L < N:
                    dx_half_real = dx_half_real.reshape(B, H, N)
                    dx_half_imag = dx_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        dx = butterfly_ifft_padded_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        dx = butterfly_ifft_padded_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    dx_half_real = dx_half_real.reshape(B, H, 32, 4096)
                    dx_half_imag = dx_half_imag.reshape(B, H, 32, 4096)

                    if x.dtype == torch.float16:
                        dx_half = butterfly_ifft_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        dx_half = butterfly_ifft_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    dx = dx_half.reshape(B, H, N)

                dk_f = torch.fft.ifft(
                    torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 16, 16, 16).transpose(-1, -2).reshape(H, 32, 16, 256).transpose(-1, -2).reshape(H, 32, 4096).transpose(-1, -2).reshape(H, N) * 32,
                    norm='forward', n=N
                ).real[..., :k_len]
            else:
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_forward(
                            dout,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                            dout,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                else:
                    x = u.reshape(B, H, 16, 8192)
                    dout = dout.reshape(B, H, 16, 8192)

                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_forward(
                            dout,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_bf16_forward(
                            dout,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 16, 8192)
                x_half_imag = x_half_imag.reshape(B, H * 16, 8192)

                dout_half_real = dout_half_real.reshape(B, H * 16, 8192)
                dout_half_imag = dout_half_imag.reshape(B, H * 16, 8192)

                dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_16_16_complex(
                    dout_half_real, dout_half_imag,
                    x_half_real, x_half_imag, k_f_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_32_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_32_ifft,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_32_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    8192, 8192
                )

                if L < N:
                    dx_half_real = dx_half_real.reshape(B, H, N)
                    dx_half_imag = dx_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        dx = butterfly_ifft_padded_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        dx = butterfly_ifft_padded_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    dx_half_real = dx_half_real.reshape(B, H, 16, 8192)
                    dx_half_imag = dx_half_imag.reshape(B, H, 16, 8192)

                    if x.dtype == torch.float16:
                        dx_half = butterfly_ifft_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        dx_half = butterfly_ifft_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    dx = dx_half.reshape(B, H, N)

                dk_f = torch.fft.ifft(
                    torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 32, 16, 16).transpose(-1, -2).reshape(H, 16, 32, 256).transpose(-1, -2).reshape(H, 16, 8192).transpose(-1, -2).reshape(H, N) * 16,
                    norm='forward', n=N
                ).real[..., :k_len]

            return dx[..., :L], dk_f, None
        elif fftconv_data.seqlen == 16 * 16384:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_forward(
                            dout,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                            dout,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            8192
                        )
                else:
                    x = u.reshape(B, H, 32, 8192)
                    dout = dout.reshape(B, H, 32, 8192)

                    
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_forward(
                            dout,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_bf16_forward(
                            dout,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                
                x_half_real = x_half_real.reshape(B, H * 32, 8192)
                x_half_imag = x_half_imag.reshape(B, H * 32, 8192)

                dout_half_real = dout_half_real.reshape(B, H * 32, 8192)
                dout_half_imag = dout_half_imag.reshape(B, H * 32, 8192)

                dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_16_16_complex(
                    dout_half_real, dout_half_imag,
                    x_half_real, x_half_imag, k_f_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_32_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_32_ifft,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_32_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    8192, 8192
                )

                if L < N:
                    dx_half_real = dx_half_real.reshape(B, H, N)
                    dx_half_imag = dx_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        dx = butterfly_ifft_padded_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        dx = butterfly_ifft_padded_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    dx_half_real = dx_half_real.reshape(B, H, 32, 8192)
                    dx_half_imag = dx_half_imag.reshape(B, H, 32, 8192)

                    if x.dtype == torch.float16:
                        dx_half = butterfly_ifft_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        dx_half = butterfly_ifft_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    dx = dx_half.reshape(B, H, N)

                dk_f = torch.fft.ifft(
                    torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 32, 16, 16).transpose(-1, -2).reshape(H, 32, 32, 256).transpose(-1, -2).reshape(H, 32, 8192).transpose(-1, -2).reshape(H, N) * 32,
                    norm='forward', n=N
                ).real[..., :k_len]
            else:
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_forward(
                            dout,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                            dout,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                else:
                    x = u.reshape(B, H, 16, 16384)
                    dout = dout.reshape(B, H, 16, 16384)

                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_forward(
                            dout,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_bf16_forward(
                            dout,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 16, 16384)
                x_half_imag = x_half_imag.reshape(B, H * 16, 16384)

                dout_half_real = dout_half_real.reshape(B, H * 16, 16384)
                dout_half_imag = dout_half_imag.reshape(B, H * 16, 16384)

                dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_32_32_complex(
                    dout_half_real, dout_half_imag,
                    x_half_real, x_half_imag, k_f_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_16_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_16_ifft,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_16_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    16384, 16384
                )

                if L < N:
                    dx_half_real = dx_half_real.reshape(B, H, N)
                    dx_half_imag = dx_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        dx = butterfly_ifft_padded_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        dx = butterfly_ifft_padded_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    dx_half_real = dx_half_real.reshape(B, H, 16, 16384)
                    dx_half_imag = dx_half_imag.reshape(B, H, 16, 16384)

                    if x.dtype == torch.float16:
                        dx_half = butterfly_ifft_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        dx_half = butterfly_ifft_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    dx = dx_half.reshape(B, H, N)

                dk_f = torch.fft.ifft(
                    torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 16, 32, 32).transpose(-1, -2).reshape(H, 16, 16, 1024).transpose(-1, -2).reshape(H, 16, 16384).transpose(-1, -2).reshape(H, N) * 16,
                    norm='forward', n=N
                ).real[..., :k_len]

            return dx[..., :L], dk_f, None
        elif fftconv_data.seqlen == 16 * 32768:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_forward(
                            dout,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                            dout,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            16384
                        )
                else:
                    x = u.reshape(B, H, 32, 16384)
                    dout = dout.reshape(B, H, 32, 16384)

                    
                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_forward(
                            dout,
                            fftconv_data.f_32_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_bf16_forward(
                            dout,
                            fftconv_data.f_32_fft_real,
                            fftconv_data.f_32_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                
                x_half_real = x_half_real.reshape(B, H * 32, 16384)
                x_half_imag = x_half_imag.reshape(B, H * 32, 16384)

                dout_half_real = dout_half_real.reshape(B, H * 32, 16384)
                dout_half_imag = dout_half_imag.reshape(B, H * 32, 16384)

                dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_32_32_complex(
                    dout_half_real, dout_half_imag,
                    x_half_real, x_half_imag, k_f_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_16_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_16_ifft,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_16_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    16384, 16384
                )

                if L < N:
                    dx_half_real = dx_half_real.reshape(B, H, N)
                    dx_half_imag = dx_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        dx = butterfly_ifft_padded_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        dx = butterfly_ifft_padded_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    dx_half_real = dx_half_real.reshape(B, H, 32, 16384)
                    dx_half_imag = dx_half_imag.reshape(B, H, 32, 16384)

                    if x.dtype == torch.float16:
                        dx_half = butterfly_ifft_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        dx_half = butterfly_ifft_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_32_ifft_real,
                            fftconv_data.f_32_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    dx = dx_half.reshape(B, H, N)

                dk_f = torch.fft.ifft(
                    torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 16, 32, 32).transpose(-1, -2).reshape(H, 32, 16, 1024).transpose(-1, -2).reshape(H, 32, 16384).transpose(-1, -2).reshape(H, N) * 32,
                    norm='forward', n=N
                ).real[..., :k_len]
            else:
                if L < N:
                    if u.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_padded_forward(
                            u,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            32768
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_forward(
                            dout,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            32768
                        )
                    else:
                        x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                            u,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            32768
                        )
                        dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                            dout,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag,
                            32768
                        )
                else:
                    x = u.reshape(B, H, 16, 32768)
                    dout = dout.reshape(B, H, 16, 32768)

                    if x.dtype == torch.float16:
                        x_half_real, x_half_imag = butterfly_forward(
                            x,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_forward(
                            dout,
                            fftconv_data.f_16_fft,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        x_half_real, x_half_imag = butterfly_bf16_forward(
                            x,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )
                        dout_half_real, dout_half_imag = butterfly_bf16_forward(
                            dout,
                            fftconv_data.f_16_fft_real,
                            fftconv_data.f_16_fft_imag,
                            fftconv_data.twiddle_factors_fft_real,
                            fftconv_data.twiddle_factors_fft_imag
                        )

                x_half_real = x_half_real.reshape(B, H * 16, 32768)
                x_half_imag = x_half_imag.reshape(B, H * 16, 32768)

                dout_half_real = dout_half_real.reshape(B, H * 16, 32768)
                dout_half_imag = dout_half_imag.reshape(B, H * 16, 32768)

                dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                    dout_half_real, dout_half_imag,
                    x_half_real, x_half_imag, k_f_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_32_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_32_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    32768, 32768
                )

                if L < N:
                    dx_half_real = dx_half_real.reshape(B, H, N)
                    dx_half_imag = dx_half_imag.reshape(B, H, N)

                    if u.dtype == torch.float16:
                        dx = butterfly_ifft_padded_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                    else:
                        dx = butterfly_ifft_padded_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag,
                            L
                        )
                else:
                    dx_half_real = dx_half_real.reshape(B, H, 16, 32768)
                    dx_half_imag = dx_half_imag.reshape(B, H, 16, 32768)

                    if x.dtype == torch.float16:
                        dx_half = butterfly_ifft_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )
                    elif x.dtype == torch.bfloat16:
                        dx_half = butterfly_ifft_bf16_forward(
                            dx_half_real, dx_half_imag,
                            fftconv_data.f_16_ifft_real,
                            fftconv_data.f_16_ifft_imag,
                            fftconv_data.twiddle_factors_ifft_real,
                            fftconv_data.twiddle_factors_ifft_imag
                        )

                    dx = dx_half.reshape(B, H, N)

                dk_f = torch.fft.ifft(
                    torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 32, 32, 32).transpose(-1, -2).reshape(H, 16, 32, 1024).transpose(-1, -2).reshape(H, 16, 32768).transpose(-1, -2).reshape(H, N) * 16,
                    norm='forward', n=N
                ).real[..., :k_len]

            return dx[..., :L], dk_f, None
        elif fftconv_data.seqlen == 32 * 32768:
            N = fftconv_data.N

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_forward(
                        dout,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                        dout,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
            else:
                x = u.reshape(B, H, 32, 32768)
                dout = dout.reshape(B, H, 32, 32768)

                
                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_forward(
                        dout,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_bf16_forward(
                        dout,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
            
            x_half_real = x_half_real.reshape(B, H * 32, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 32, 32768)

            dout_half_real = dout_half_real.reshape(B, H * 32, 32768)
            dout_half_imag = dout_half_imag.reshape(B, H * 32, 32768)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                dout_half_real, dout_half_imag,
                x_half_real, x_half_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            if L < N:
                dx_half_real = dx_half_real.reshape(B, H, N)
                dx_half_imag = dx_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    dx = butterfly_ifft_padded_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    dx = butterfly_ifft_padded_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                dx_half_real = dx_half_real.reshape(B, H, 32, 32768)
                dx_half_imag = dx_half_imag.reshape(B, H, 32, 32768)

                if x.dtype == torch.float16:
                    dx_half = butterfly_ifft_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    dx_half = butterfly_ifft_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                dx = dx_half.reshape(B, H, N)

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 32, 32, 32).transpose(-1, -2).reshape(H, 32, 32, 1024).transpose(-1, -2).reshape(H, 32, 32768).transpose(-1, -2).reshape(H, N) * 32,
                norm='forward', n=N
            ).real[..., :k_len]

            return dx[..., :L], dk_f, None
        elif fftconv_data.seqlen == 64 * 32768:
            N = fftconv_data.N

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_64_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_forward(
                        dout,
                        fftconv_data.f_64_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_64_fft_real,
                        fftconv_data.f_64_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                        dout,
                        fftconv_data.f_64_fft_real,
                        fftconv_data.f_64_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
            else:
                x = u.reshape(B, H, 64, 32768)
                dout = dout.reshape(B, H, 64, 32768)

                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_64_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_forward(
                        dout,
                        fftconv_data.f_64_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_64_fft_real,
                        fftconv_data.f_64_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_bf16_forward(
                        dout,
                        fftconv_data.f_64_fft_real,
                        fftconv_data.f_64_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )

            x_half_real = x_half_real.reshape(B, H * 64, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 64, 32768)

            dout_half_real = dout_half_real.reshape(B, H * 64, 32768)
            dout_half_imag = dout_half_imag.reshape(B, H * 64, 32768)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                dout_half_real, dout_half_imag,
                x_half_real, x_half_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            if L < N:
                dx_half_real = dx_half_real.reshape(B, H, N)
                dx_half_imag = dx_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    dx = butterfly_ifft_padded_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_64_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    dx = butterfly_ifft_padded_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_64_ifft_real,
                        fftconv_data.f_64_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                dx_half_real = dx_half_real.reshape(B, H, 64, 32768)
                dx_half_imag = dx_half_imag.reshape(B, H, 64, 32768)

                if x.dtype == torch.float16:
                    dx_half = butterfly_ifft_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_64_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    dx_half = butterfly_ifft_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_64_ifft_real,
                        fftconv_data.f_64_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                dx = dx_half.reshape(B, H, N)

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 64, 32, 32, 32).transpose(-1, -2).reshape(H, 64, 32, 1024).transpose(-1, -2).reshape(H, 64, 32768).transpose(-1, -2).reshape(H, N) * 64,
                norm='forward', n=N
            ).real[..., :k_len]

            return dx[..., :L], dk_f, None
        elif fftconv_data.seqlen == 128 * 32768:
            N = fftconv_data.N

            # assert(N == L)
            if L < N:
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_forward(
                        u,
                        fftconv_data.f_128_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_forward(
                        dout,
                        fftconv_data.f_128_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_bf16_forward(
                        u,
                        fftconv_data.f_128_fft_real,
                        fftconv_data.f_128_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
                    dout_half_real, dout_half_imag = butterfly_padded_bf16_forward(
                        dout,
                        fftconv_data.f_128_fft_real,
                        fftconv_data.f_128_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        32768
                    )
            else:
                x = u.reshape(B, H, 128, 32768)
                dout = dout.reshape(B, H, 128, 32768)

                if x.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_forward(
                        x,
                        fftconv_data.f_128_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_forward(
                        dout,
                        fftconv_data.f_128_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    x_half_real, x_half_imag = butterfly_bf16_forward(
                        x,
                        fftconv_data.f_128_fft_real,
                        fftconv_data.f_128_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )
                    dout_half_real, dout_half_imag = butterfly_bf16_forward(
                        dout,
                        fftconv_data.f_128_fft_real,
                        fftconv_data.f_128_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag
                    )

            x_half_real = x_half_real.reshape(B, H * 128, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 128, 32768)

            dout_half_real = dout_half_real.reshape(B, H * 128, 32768)
            dout_half_imag = dout_half_imag.reshape(B, H * 128, 32768)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                dout_half_real, dout_half_imag,
                x_half_real, x_half_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            if L < N:
                dx_half_real = dx_half_real.reshape(B, H, N)
                dx_half_imag = dx_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    dx = butterfly_ifft_padded_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_128_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
                else:
                    dx = butterfly_ifft_padded_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_128_ifft_real,
                        fftconv_data.f_128_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L
                    )
            else:
                dx_half_real = dx_half_real.reshape(B, H, 128, 32768)
                dx_half_imag = dx_half_imag.reshape(B, H, 128, 32768)

                if x.dtype == torch.float16:
                    dx_half = butterfly_ifft_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_128_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )
                elif x.dtype == torch.bfloat16:
                    dx_half = butterfly_ifft_bf16_forward(
                        dx_half_real, dx_half_imag,
                        fftconv_data.f_128_ifft_real,
                        fftconv_data.f_128_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag
                    )

                dx = dx_half.reshape(B, H, N)

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 128, 32, 32, 32).transpose(-1, -2).reshape(H, 128, 32, 1024).transpose(-1, -2).reshape(H, 128, 32768).transpose(-1, -2).reshape(H, N) * 128,
                norm='forward', n=N
            ).real[..., :k_len]

            return dx[..., :L], dk_f, None
        else:
            raise NotImplementedError(f'seqlen {fftconv_data.seqlen} not supported for FlashFFTConv bwd')

class GatedFlashFFTConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, fftconv_data, pregate, postgate):
        # assert(u.dtype == fftconv_data.dtype)

        B, H, L = u.shape

        if fftconv_data.seqlen in [512, 2048]:
            k_f = torch.fft.rfft(k, n=fftconv_data.seqlen)
        else:
            k_f = torch.fft.fft(k, n=fftconv_data.seqlen)

        ctx.fftconv_data = fftconv_data
        ctx.k_len = k.shape[-1]

        if fftconv_data.seqlen in [256, 1024]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted, pregate, postgate)

            return monarch_conv_forward(
                u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                pregate, postgate,
                N, L, sqrt_N
            )
        elif fftconv_data.seqlen in [512, 2048]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N

            k_f = torch.view_as_real(k_f).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f, pregate, postgate)

            return monarch_conv_forward_r2r(
                u, k_f,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.twid,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                pregate, postgate,
                N, L, sqrt_N
            )
        elif fftconv_data.seqlen == 4096:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N
            sqrt_N_256 = fftconv_data.sqrt_N_256

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, sqrt_N_256, sqrt_N).transpose(-1, -2).reshape(H, sqrt_N, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted, pregate, postgate)

            out = monarch_conv_forward_16_16_16(
                u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft,
                fftconv_data.twiddle_factors_fft_16_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_sqrt_N_ifft,
                fftconv_data.twiddle_factors_ifft_16_256, fftconv_data.twiddle_factors_ifft_16_16,
                pregate, postgate,
                N, L, sqrt_N_256, sqrt_N
            )

            return out
        elif fftconv_data.seqlen == 8192:
            N = fftconv_data.N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, 256, 32).transpose(-1, -2).reshape(H, 32, 16, 16).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted, pregate, postgate)

            return monarch_conv_forward_32_16_16(
                u, k_f_permuted,
                fftconv_data.f_32_fft, fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_32_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_32_ifft, fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_32_256, fftconv_data.twiddle_factors_ifft_16_16,
                pregate, postgate,
                N, L
            )
        elif fftconv_data.seqlen == 16384:
            N = fftconv_data.N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, 1024, 16).transpose(-1, -2).reshape(H, 16, 32, 32).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted, pregate, postgate)

            return monarch_conv_forward_16_32_32(
                u, k_f_permuted,
                fftconv_data.f_16_fft, fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_16_1K, fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_16_ifft, fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_16_1K, fftconv_data.twiddle_factors_ifft_32_32,
                pregate, postgate,
                N, L
            )
        elif fftconv_data.seqlen == 32768:
            N = fftconv_data.N

            # assert(L == N)
            k_f_permuted = torch.view_as_real(k_f.reshape(H, 1024, 32).transpose(-1, -2).reshape(H, 32, 32, 32).transpose(-1, -2).reshape(H, N)).to(fftconv_data.dtype).contiguous()

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_permuted, pregate, postgate)

            return monarch_conv_forward_32_32_32(
                u, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                pregate, postgate,
                N, L
            )
        if fftconv_data.seqlen == 16 * 4096:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 4096, 16).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 16, 256, 16).transpose(-1, -2).reshape(H, 16, 16, 16, 16).transpose(-1, -2).reshape(H * 16, 4096)).contiguous().to(fftconv_data.dtype)

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

            if u.dtype == torch.float16:
                x_half_real, x_half_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    pregate
                )
            else:
                x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_16_fft_real,
                    fftconv_data.f_16_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    pregate
                )

            x_half_real = x_half_real.reshape(B, H * 16, 4096)
            x_half_imag = x_half_imag.reshape(B, H * 16, 4096)

            out_half_real, out_half_imag = monarch_conv_forward_16_16_16_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            out_half_real = out_half_real.reshape(B, H, N)
            out_half_imag = out_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                x = butterfly_ifft_padded_gated_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )
            else:
                x = butterfly_ifft_padded_gated_bf16_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_16_ifft_real,
                    fftconv_data.f_16_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )
            
            return x[..., :L]
        if fftconv_data.seqlen == 16 * 8192:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:
                k_f_permuted = k_f.reshape(H, 4096, 32).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 256, 16).transpose(-1, -2).reshape(H, 32, 16, 16, 16).transpose(-1, -2).reshape(H * 32, 4096)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

                # assert(N == L)
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_gated_forward(
                        u,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096,
                        pregate
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                        u,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        4096,
                        pregate
                    )

                x_half_real = x_half_real.reshape(B, H * 32, 4096)
                x_half_imag = x_half_imag.reshape(B, H * 32, 4096)

                out_half_real, out_half_imag = monarch_conv_forward_16_16_16_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_16_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_16_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    4096, 4096
                )

                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_gated_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L,
                        postgate
                    )
                else:
                    x = butterfly_ifft_padded_gated_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L,
                        postgate
                    )
            else:
                raise NotImplementedError

            return x[..., :L]
        elif fftconv_data.seqlen == 16 * 16384:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:

                k_f_permuted = k_f.reshape(H, 8192, 32).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 256, 32).transpose(-1, -2).reshape(H, 32, 32, 16, 16).transpose(-1, -2).reshape(H * 32, 8192)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_gated_forward(
                        u,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        8192,
                        pregate
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                        u,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        8192,
                        pregate
                    )

                x_half_real = x_half_real.reshape(B, H * 32, 8192)
                x_half_imag = x_half_imag.reshape(B, H * 32, 8192)

                out_half_real, out_half_imag = monarch_conv_forward_32_16_16_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_32_fft,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_32_256,
                    fftconv_data.twiddle_factors_fft_16_16,
                    fftconv_data.f_32_ifft,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_32_256,
                    fftconv_data.twiddle_factors_ifft_16_16,
                    8192, 8192
                )

                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_gated_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L,
                        postgate
                    )
                else:
                    x = butterfly_ifft_padded_gated_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L,
                        postgate
                    )
            else:
                raise NotImplementedError

            return x[..., :L]
        elif fftconv_data.seqlen == 16 * 32768:
            N = fftconv_data.N

            if fftconv_data.use_32_butterfly:
                k_f_permuted = k_f.reshape(H, 16384, 32).transpose(-1, -2).reshape(H, N)
                k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 1024, 16).transpose(-1, -2).reshape(H, 32, 16, 32, 32).transpose(-1, -2).reshape(H * 32, 16384)).contiguous().to(fftconv_data.dtype)

                if fftconv_data.training:
                    ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

                # assert(N == L)
                if u.dtype == torch.float16:
                    x_half_real, x_half_imag = butterfly_padded_gated_forward(
                        u,
                        fftconv_data.f_32_fft,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        16384,
                        pregate
                    )
                else:
                    x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                        u,
                        fftconv_data.f_32_fft_real,
                        fftconv_data.f_32_fft_imag,
                        fftconv_data.twiddle_factors_fft_real,
                        fftconv_data.twiddle_factors_fft_imag,
                        16384,
                        pregate
                    )

                x_half_real = x_half_real.reshape(B, H * 32, 16384)
                x_half_imag = x_half_imag.reshape(B, H * 32, 16384)

                out_half_real, out_half_imag = monarch_conv_forward_16_32_32_complex(
                    x_half_real, x_half_imag, k_f_double_permuted,
                    fftconv_data.f_16_fft,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_16_1K,
                    fftconv_data.twiddle_factors_fft_32_32,
                    fftconv_data.f_16_ifft,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_16_1K,
                    fftconv_data.twiddle_factors_ifft_32_32,
                    16384, 16384
                )

                out_half_real = out_half_real.reshape(B, H, N)
                out_half_imag = out_half_imag.reshape(B, H, N)

                if u.dtype == torch.float16:
                    x = butterfly_ifft_padded_gated_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L,
                        postgate
                    )
                else:
                    x = butterfly_ifft_padded_gated_bf16_forward(
                        out_half_real, out_half_imag,
                        fftconv_data.f_32_ifft_real,
                        fftconv_data.f_32_ifft_imag,
                        fftconv_data.twiddle_factors_ifft_real,
                        fftconv_data.twiddle_factors_ifft_imag,
                        L,
                        postgate
                    )
            else:
                raise NotImplementedError

            return x[..., :L]
        elif fftconv_data.seqlen == 32 * 32768:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 32768, 32).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 32, 1024, 32).transpose(-1, -2).reshape(H, 32, 32, 32, 32).transpose(-1, -2).reshape(H * 32, 32768)).contiguous().to(fftconv_data.dtype)
            
            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

            # assert(N == L)
            if u.dtype == torch.float16:
                x_half_real, x_half_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
            else:
                x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                
            x_half_real = x_half_real.reshape(B, H * 32, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 32, 32768)

            out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            out_half_real = out_half_real.reshape(B, H, N)
            out_half_imag = out_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                x = butterfly_ifft_padded_gated_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )
            else:
                x = butterfly_ifft_padded_gated_bf16_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )

            return x[..., :L]
        elif fftconv_data.seqlen == 64 * 32768:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 32768, 64).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 64, 1024, 32).transpose(-1, -2).reshape(H, 64, 32, 32, 32).transpose(-1, -2).reshape(H * 64, 32768)).contiguous().to(fftconv_data.dtype)

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

            # assert(N == L)
            if u.dtype == torch.float16:
                x_half_real, x_half_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_64_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
            else:
                x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_64_fft_real,
                    fftconv_data.f_64_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )

            x_half_real = x_half_real.reshape(B, H * 64, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 64, 32768)

            out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            out_half_real = out_half_real.reshape(B, H, N)
            out_half_imag = out_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                x = butterfly_ifft_padded_gated_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_64_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )
            else:
                x = butterfly_ifft_padded_gated_bf16_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_64_ifft_real,
                    fftconv_data.f_64_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )

            return x[..., :L]    
        elif fftconv_data.seqlen == 128 * 32768:
            N = fftconv_data.N

            k_f_permuted = k_f.reshape(H, 32768, 128).transpose(-1, -2).reshape(H, N)
            k_f_double_permuted = torch.view_as_real(k_f_permuted.reshape(H, 128, 1024, 32).transpose(-1, -2).reshape(H, 128, 32, 32, 32).transpose(-1, -2).reshape(H * 128, 32768)).contiguous().to(fftconv_data.dtype)

            if fftconv_data.training:
                ctx.save_for_backward(u, k_f_double_permuted, pregate, postgate)

            # assert(N == L)
            if u.dtype == torch.float16:
                x_half_real, x_half_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_128_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
            else:
                x_half_real, x_half_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_128_fft_real,
                    fftconv_data.f_128_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )

            x_half_real = x_half_real.reshape(B, H * 128, 32768)
            x_half_imag = x_half_imag.reshape(B, H * 128, 32768)

            out_half_real, out_half_imag = monarch_conv_forward_32_32_32_complex(
                x_half_real, x_half_imag, k_f_double_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            out_half_real = out_half_real.reshape(B, H, N)
            out_half_imag = out_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                x = butterfly_ifft_padded_gated_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_128_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )
            else:
                x = butterfly_ifft_padded_gated_bf16_forward(
                    out_half_real, out_half_imag,
                    fftconv_data.f_128_ifft_real,
                    fftconv_data.f_128_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    postgate
                )

            return x[..., :L]
        else:
            raise NotImplementedError(f'seqlen {fftconv_data.seqlen} not supported for GatedFlashFFTConv fwd')

    @staticmethod
    def backward(ctx, dout):
        fftconv_data = ctx.fftconv_data
        # assert(dout.dtype == fftconv_data.dtype)

        B, H, L = dout.shape
        dout = dout.contiguous()

        u, k_f_permuted, pregate, postgate = ctx.saved_tensors
        k_len = ctx.k_len
        
        if fftconv_data.seqlen in [256, 1024]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N

            du, dk_f_permuted, dpregate, dpostgate = monarch_conv_backward(
                dout, u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                pregate, postgate,
                N, L, sqrt_N
            )
            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None, dpregate, dpostgate
        elif fftconv_data.seqlen in [512, 2048]:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N
            
            du, dk_f, dpregate, dpostgate = monarch_conv_backward_r2r(
                dout, u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft, fftconv_data.twiddle_factors_fft,
                fftconv_data.twid,
                fftconv_data.f_sqrt_N_ifft, fftconv_data.twiddle_factors_ifft,
                pregate, postgate,
                N, L, sqrt_N
            )
            dk_f = torch.fft.irfft(
                torch.view_as_complex(dk_f.to(torch.float32)), n=fftconv_data.seqlen, norm='forward'
            ).real[..., :k_len] / 2

            return du, dk_f, None, dpregate, dpostgate
        elif fftconv_data.seqlen == 4096:
            N = fftconv_data.N
            sqrt_N = fftconv_data.sqrt_N
            sqrt_N_256 = fftconv_data.sqrt_N_256

            du, dk_f_permuted, dpregate, dpostgate = monarch_conv_backward_16_16_16(
                dout, u, k_f_permuted,
                fftconv_data.f_sqrt_N_fft,
                fftconv_data.twiddle_factors_fft_16_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_sqrt_N_ifft,
                fftconv_data.twiddle_factors_ifft_16_256, fftconv_data.twiddle_factors_ifft_16_16,
                pregate, postgate,
                N, L, sqrt_N_256, sqrt_N
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, sqrt_N, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, sqrt_N, sqrt_N_256).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None, dpregate, dpostgate
        elif fftconv_data.seqlen == 8192:
            N = fftconv_data.N

            du, dk_f_permuted, dpregate, dpostgate = monarch_conv_backward_32_16_16(
                dout, u, k_f_permuted,
                fftconv_data.f_32_fft, fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_32_256, fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_32_ifft, fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_32_256, fftconv_data.twiddle_factors_ifft_16_16,
                pregate, postgate,
                N, L
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 16, 16).transpose(-1, -2).reshape(H, 32, 256).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None, dpregate, dpostgate
        elif fftconv_data.seqlen == 16384:
            N = fftconv_data.N

            du, dk_f_permuted, dpregate, dpostgate = monarch_conv_backward_16_32_32(
                dout, u, k_f_permuted,
                fftconv_data.f_16_fft, fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_16_1K, fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_16_ifft, fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_16_1K, fftconv_data.twiddle_factors_ifft_32_32,
                pregate, postgate,
                N, L
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 32, 32).transpose(-1, -2).reshape(H, 16, 1024).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None, dpregate, dpostgate
        elif fftconv_data.seqlen == 32768:
            N = fftconv_data.N

            du, dk_f_permuted, dpregate, dpostgate = monarch_conv_backward_32_32_32(
                dout, u, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                pregate, postgate,
                N, L
            )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 32, 32).transpose(-1, -2).reshape(H, 32, 1024).transpose(-1, -2).reshape(H, N),
                norm='forward', n=N
            ).real[..., :k_len]

            return du, dk_f, None, dpregate, dpostgate
        elif fftconv_data.seqlen == 16 * 4096:
            N = fftconv_data.N

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_16_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_16_fft_real,
                    fftconv_data.f_16_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_16_fft_real,
                    fftconv_data.f_16_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 16, 4096)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 16, 4096)

            y_half_real, y_half_imag = monarch_conv_forward_16_16_16_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_16_ifft_real,
                    fftconv_data.f_16_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 16, 4096)
            dout_half_imag = dout_half_imag.reshape(B, H * 16, 4096)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_16_16_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_16_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_16_ifft_real,
                    fftconv_data.f_16_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_16_ifft_real,
                    fftconv_data.f_16_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 16, 16, 16, 16).transpose(-1, -2).reshape(H, 16, 16, 256).transpose(-1, -2).reshape(H, 16, 4096).transpose(-1, -2).reshape(H, N) * 16,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        elif fftconv_data.seqlen == 16 * 8192:
            N = fftconv_data.N
            assert fftconv_data.use_32_butterfly

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    4096,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 32, 4096)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 32, 4096)

            y_half_real, y_half_imag = monarch_conv_forward_16_16_16_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 32, 4096)
            dout_half_imag = dout_half_imag.reshape(B, H * 32, 4096)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_16_16_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_16_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_16_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                4096, 4096
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 16, 16, 16).transpose(-1, -2).reshape(H, 32, 16, 256).transpose(-1, -2).reshape(H, 32, 4096).transpose(-1, -2).reshape(H, N) * 32,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        elif fftconv_data.seqlen == 16 * 16384:
            N = fftconv_data.N
            assert fftconv_data.use_32_butterfly

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    8192,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    8192,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    8192,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    8192,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 32, 8192)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 32, 8192)

            y_half_real, y_half_imag = monarch_conv_forward_32_16_16_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_32_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_32_ifft,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_32_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                8192, 8192
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 32, 8192)
            dout_half_imag = dout_half_imag.reshape(B, H * 32, 8192)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_16_16_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.f_16_fft,
                fftconv_data.twiddle_factors_fft_32_256,
                fftconv_data.twiddle_factors_fft_16_16,
                fftconv_data.f_32_ifft,
                fftconv_data.f_16_ifft,
                fftconv_data.twiddle_factors_ifft_32_256,
                fftconv_data.twiddle_factors_ifft_16_16,
                8192, 8192
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 32, 16, 16).transpose(-1, -2).reshape(H, 32, 32, 256).transpose(-1, -2).reshape(H, 32, 8192).transpose(-1, -2).reshape(H, N) * 32,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        elif fftconv_data.seqlen == 16 * 32768:
            N = fftconv_data.N
            assert fftconv_data.use_32_butterfly

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    16384,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    16384,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    16384,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    16384,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 32, 16384)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 32, 16384)

            y_half_real, y_half_imag = monarch_conv_forward_16_32_32_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_16_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_16_ifft,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_16_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                16384, 16384
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 32, 16384)
            dout_half_imag = dout_half_imag.reshape(B, H * 32, 16384)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_16_32_32_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_16_fft,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_16_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_16_ifft,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_16_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                16384, 16384
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 16, 32, 32).transpose(-1, -2).reshape(H, 32, 16, 1024).transpose(-1, -2).reshape(H, 32, 16384).transpose(-1, -2).reshape(H, N) * 32,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        elif fftconv_data.seqlen == 32 * 32768:
            N = fftconv_data.N

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_32_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_32_fft_real,
                    fftconv_data.f_32_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 32, 32768)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 32, 32768)

            y_half_real, y_half_imag = monarch_conv_forward_32_32_32_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 32, 32768)
            dout_half_imag = dout_half_imag.reshape(B, H * 32, 32768)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_32_ifft_real,
                    fftconv_data.f_32_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 32, 32, 32, 32).transpose(-1, -2).reshape(H, 32, 32, 1024).transpose(-1, -2).reshape(H, 32, 32768).transpose(-1, -2).reshape(H, N) * 32,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        elif fftconv_data.seqlen == 64 * 32768:
            N = fftconv_data.N

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_64_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_64_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_64_fft_real,
                    fftconv_data.f_64_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_64_fft_real,
                    fftconv_data.f_64_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 64, 32768)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 64, 32768)

            y_half_real, y_half_imag = monarch_conv_forward_32_32_32_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_64_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_64_ifft_real,
                    fftconv_data.f_64_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 64, 32768)
            dout_half_imag = dout_half_imag.reshape(B, H * 64, 32768)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_64_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_64_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_64_ifft_real,
                    fftconv_data.f_64_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_64_ifft_real,
                    fftconv_data.f_64_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 64, 32, 32, 32).transpose(-1, -2).reshape(H, 64, 32, 1024).transpose(-1, -2).reshape(H, 64, 32768).transpose(-1, -2).reshape(H, N) * 64,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        elif fftconv_data.seqlen == 128 * 32768:
            N = fftconv_data.N

            if u.dtype == torch.float16:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_forward(
                    u,
                    fftconv_data.f_128_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_forward(
                    dout,
                    fftconv_data.f_128_fft,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    postgate
                )
            else:
                u_gate1_real, u_gate1_imag = butterfly_padded_gated_bf16_forward(
                    u,
                    fftconv_data.f_128_fft_real,
                    fftconv_data.f_128_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    pregate
                )
                dout_half_real, dout_half_imag = butterfly_padded_gated_bf16_forward(
                    dout,
                    fftconv_data.f_128_fft_real,
                    fftconv_data.f_128_fft_imag,
                    fftconv_data.twiddle_factors_fft_real,
                    fftconv_data.twiddle_factors_fft_imag,
                    32768,
                    postgate
                )

            u_gate1_real = u_gate1_real.reshape(B, H * 128, 32768)
            u_gate1_imag = u_gate1_imag.reshape(B, H * 128, 32768)

            y_half_real, y_half_imag = monarch_conv_forward_32_32_32_complex(
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            y_half_real = y_half_real.reshape(B, H, N)
            y_half_imag = y_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                dpostgate = butterfly_ifft_padded_gated_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_128_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )
            else:
                dpostgate = butterfly_ifft_padded_gated_bf16_forward(
                    y_half_real, y_half_imag,
                    fftconv_data.f_128_ifft_real,
                    fftconv_data.f_128_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    dout
                )

            dout_half_real = dout_half_real.reshape(B, H * 128, 32768)
            dout_half_imag = dout_half_imag.reshape(B, H * 128, 32768)

            dx_half_real, dx_half_imag, dk_f_permuted = monarch_conv_backward_32_32_32_complex(
                dout_half_real, dout_half_imag,
                u_gate1_real, u_gate1_imag, k_f_permuted,
                fftconv_data.f_32_fft,
                fftconv_data.twiddle_factors_fft_32_1K,
                fftconv_data.twiddle_factors_fft_32_32,
                fftconv_data.f_32_ifft,
                fftconv_data.twiddle_factors_ifft_32_1K,
                fftconv_data.twiddle_factors_ifft_32_32,
                32768, 32768
            )

            dx_half_real = dx_half_real.reshape(B, H, N)
            dx_half_imag = dx_half_imag.reshape(B, H, N)

            if u.dtype == torch.float16:
                du = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_128_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_128_ifft,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )
            else:
                du = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_128_ifft_real,
                    fftconv_data.f_128_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    pregate
                )
                dpregate = butterfly_ifft_padded_gated_bf16_forward(
                    dx_half_real, dx_half_imag,
                    fftconv_data.f_128_ifft_real,
                    fftconv_data.f_128_ifft_imag,
                    fftconv_data.twiddle_factors_ifft_real,
                    fftconv_data.twiddle_factors_ifft_imag,
                    L,
                    u
                )

            dk_f = torch.fft.ifft(
                torch.view_as_complex(dk_f_permuted.to(torch.float32)).reshape(H, 128, 32, 32, 32).transpose(-1, -2).reshape(H, 128, 32, 1024).transpose(-1, -2).reshape(H, 128, 32768).transpose(-1, -2).reshape(H, N) * 128,
                norm='forward', n=N
            ).real[..., :k_len]

            return du[..., :L], dk_f, None, dpregate[..., :L], dpostgate[..., :L]
        else:
            raise NotImplementedError(f'seqlen {fftconv_data.seqlen} not supported for GatedFlashFFTConv bwd')