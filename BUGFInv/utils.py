"""
Utility functions module
"""

import torch
import warnings
from typing import Optional

MAX_SIZE = 10000

def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=None):
    """
    Safely compute Cholesky decomposition with automatic jitter for non-positive definite matrices
    
    Args:
        A (Tensor): Tensor to compute Cholesky decomposition of
        upper (bool, optional): Whether to return upper triangular matrix
        out (Tensor, optional): Output tensor
        jitter (float, optional): Jitter value to add to diagonal
        max_tries (int, optional): Maximum number of attempts
    """
    L = _psd_safe_cholesky(A, out=out, jitter=jitter, max_tries=max_tries)
    if torch.any(torch.isnan(L)):
        L = _psd_safe_cholesky(A.cpu(), out=out,
                               jitter=jitter, max_tries=max_tries).to(A.device)

    if upper:
        if out is not None:
            out = out.transpose_(-1, -2)
        else:
            L = L.mT
    return L


def _psd_safe_cholesky(A, out=None, jitter=None, max_tries=None):
    """
    Internal implementation of Cholesky decomposition
    """
    
    L, info = torch.linalg.cholesky_ex(A, out=out)

    if not torch.any(info):
        return L

    isnan = torch.isnan(A)
    if isnan.any():
        warnings.warn(f"cholesky_cpu: {isnan.sum().item()} of "
                      f"{A.numel()} elements of the {A.shape} tensor are NaN.")
        exit(0)

    if jitter is None:
        jitter = 1e-7 if A.dtype == torch.float32 else 1e-9
    if max_tries is None:
        max_tries = 25
        
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # Add jitter only where needed
        diag_add = ((info > 0) * (jitter_new - jitter_prev)).unsqueeze(-1).expand(*Aprime.shape[:-1])
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        warnings.warn(f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal")
        L, info = torch.linalg.cholesky_ex(Aprime, out=out)

        if not torch.any(info):
            return L
    raise ValueError(f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}.")