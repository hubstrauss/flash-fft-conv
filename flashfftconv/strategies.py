from abc import ABC, abstractmethod
import torch
import math

from flashfftconv.utils import fft_matrix, ifft_matrix, compute_twiddle_factors_fft, compute_twiddle_factors_ifft

class InitializationStrategy(ABC):
    def __init__(self, module, dtype, use_32_butterfly):
        self.module = module
        self.dtype = dtype
        self.use_32_butterfly = use_32_butterfly

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def _register_buffers(self, buffers_dict):
        for name, tensor in buffers_dict.items():
            self.module.register_buffer(name, tensor)

    def _compute_fft_matrices(self, N):
        """Compute FFT and IFFT matrices for a given N."""
        f_fft = torch.view_as_real(fft_matrix(N)).to(self.dtype)
        f_ifft = torch.view_as_real(ifft_matrix(N)).to(self.dtype)
        return f_fft, f_ifft

    def _compute_twiddle_factors(self, N1, N2, normalize=False):
        """Compute twiddle factors with optional normalization."""
        twiddle_fft = torch.view_as_real(compute_twiddle_factors_fft(N1, N2))
        twiddle_ifft = torch.view_as_real(compute_twiddle_factors_ifft(N1, N2))
        if normalize:
            twiddle_fft /= self.module.N
        return twiddle_fft.to(self.dtype), twiddle_ifft.to(self.dtype)

class InitializationStrategyBelow4096(InitializationStrategy):
    def initialize(self):
        # Derived classes will override this method to provide specific parameters
        N, sqrt_N, twiddle_configs = self.get_parameters()

        self.module.N = N
        self.module.sqrt_N = sqrt_N

        # Compute and register FFT/IFFT matrices
        f_sqrt_N_fft, f_sqrt_N_ifft = self._compute_fft_matrices(sqrt_N)
        self._register_buffers({
            'f_sqrt_N_fft': f_sqrt_N_fft,
            'f_sqrt_N_ifft': f_sqrt_N_ifft,
        })

        # Compute and register twiddle factors
        for name, (n1, n2, normalize) in twiddle_configs.items():
            twiddle_fft, twiddle_ifft = self._compute_twiddle_factors(n1, n2, normalize)
            self._register_buffers({
                f"twiddle_factors_{name}_fft": twiddle_fft,
                f"twiddle_factors_{name}_ifft": twiddle_ifft
            })

        self.additional_initialize()

    @abstractmethod
    def get_parameters(self):
        pass

    def additional_initialize(self):
        pass

class InitializationStrategyAbove4096(InitializationStrategy):
    def initialize(self):
        """Initialize FFT/IFFT buffers and twiddle factors for seqlen above 4096."""
        N1, N2, twiddle_configs = self.get_parameters()

        # Set module attributes
        self.module.N1 = N1
        self.module.N2 = N2

        # Compute and register FFT/IFFT matrices for N1 and N2
        self._register_fft_ifft_buffers(N1, N2)

        # Compute and register twiddle factors
        for name, (n1, n2, normalize) in twiddle_configs.items():
            twiddle_fft, twiddle_ifft = self._compute_twiddle_factors(n1, n2, normalize)
            self._register_buffers({
                f"twiddle_factors_{name}_fft": twiddle_fft,
                f"twiddle_factors_{name}_ifft": twiddle_ifft
            })

    @abstractmethod
    def get_parameters(self):
        pass

    def _register_fft_ifft_buffers(self, N1, N2):
        """Compute and register FFT/IFFT buffers for N1 and N2."""
        f_N1_fft, f_N1_ifft = self._compute_fft_matrices(N1)
        f_N2_fft, f_N2_ifft = self._compute_fft_matrices(N2)
        self._register_buffers({
            f"f_{N1}_fft": f_N1_fft,
            f"f_{N1}_ifft": f_N1_ifft,
            f"f_{N2}_fft": f_N2_fft,
            f"f_{N2}_ifft": f_N2_ifft
        })

class Strategy256_1024(InitializationStrategyBelow4096):
    def get_parameters(self):
        N = self.module.seqlen
        sqrt_N = int(math.sqrt(N))
        twiddle_configs = {
            "sqrt_N": (sqrt_N, sqrt_N, True)
        }
        return N, sqrt_N, twiddle_configs
    
class Strategy512_2048(InitializationStrategyBelow4096):
    def get_parameters(self):
        N = self.module.seqlen // 2
        sqrt_N = int(math.sqrt(N))
        twiddle_configs = {
            "sqrt_N": (sqrt_N, sqrt_N, True)
        }
        return N, sqrt_N, twiddle_configs

    def additional_initialize(self):
        """Register the 'twid' buffer specific to this strategy."""
        N = self.module.N
        twid = torch.view_as_real(
            torch.exp(-2j * torch.pi * torch.arange(N) / self.module.seqlen)
        ).to(self.dtype)
        self.module.register_buffer('twid', twid)

class Strategy4096(InitializationStrategyBelow4096):
    def get_parameters(self):
        N = self.module.seqlen
        sqrt_N = 16
        sqrt_N_256 = 256
        twiddle_configs = {
            "16_16": (sqrt_N, sqrt_N, False),
            "16_256": (sqrt_N, sqrt_N_256, True)
        }
        return N, sqrt_N, twiddle_configs

class Strategy8192(InitializationStrategyAbove4096):
    def get_parameters(self):
        N1 = 32
        N2 = 16
        twiddle_configs = {
            "16_16": (N2, N2, False),
            "32_256": (N1, 256, True)
        }
        return N1, N2, twiddle_configs

class Strategy16384(InitializationStrategyAbove4096):
    def get_parameters(self):
        N1 = 16
        N2 = 32
        twiddle_configs = {
            "32_32": (N2, N2, False),
            "16_1K": (N1, 1024, True)
        }
        return N1, N2, twiddle_configs

class Strategy32768(InitializationStrategyAbove4096):
    def get_parameters(self):
        N1 = 32
        N2 = 32
        twiddle_configs = {
            "32_32": (N2, N2, False),
            "32_1K": (N2, 1024, True)
        }
        return N1, N2, twiddle_configs