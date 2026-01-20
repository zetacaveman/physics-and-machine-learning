# rmatrixnet_train.py
# PyTorch implementation of the R-mAtrIx Net "solver" training loop (YBE + regularity + optional H constraint).
#
# Requirements: torch
#
# Notes:
# - This is a clear, correct reference implementation (not highly optimized).
# - For speed, you can vectorize the batch and avoid explicit 8x8 matrices via einsum.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# -------------------------
# Utilities: permutations
# -------------------------

def permutation_swap_VV(d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Permutation (swap) operator P on V ⊗ V (dim d^2 x d^2):
      P |i>⊗|j> = |j>⊗|i>
    """
    dim = d * d
    P = torch.zeros((dim, dim), device=device, dtype=dtype)
    for i in range(d):
        for j in range(d):
            src = i * d + j
            dst = j * d + i
            P[dst, src] = 1.0
    return P


def permutation_swap_23_VVV(d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Swap operator on V ⊗ V ⊗ V (dim d^3 x d^3) that swaps factors 2 and 3:
      P23 |a>⊗|b>⊗|c> = |a>⊗|c>⊗|b>
    """
    dim = d * d * d
    P23 = torch.zeros((dim, dim), device=device, dtype=dtype)
    for a in range(d):
        for b in range(d):
            for c in range(d):
                src = (a * d + b) * d + c
                dst = (a * d + c) * d + b
                P23[dst, src] = 1.0
    return P23


def _to_c(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # Cast x to complex dtype matching "like"
    cdtype = like.dtype if torch.is_complex(like) else torch.complex64
    return x.to(dtype=cdtype)

def embed_R12(R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    I = _to_c(I, R)
    return torch.kron(R, I)

def embed_R23(R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    I = _to_c(I, R)
    return torch.kron(I, R)

def embed_R13(R: torch.Tensor, I: torch.Tensor, P23: torch.Tensor) -> torch.Tensor:
    I = _to_c(I, R)
    P23 = _to_c(P23, R)
    R12 = torch.kron(R, I)
    return P23 @ R12 @ P23


# -------------------------
# MLP building blocks
# -------------------------

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ScalarMLP(nn.Module):
    """
    Real MLP: (1 -> width -> width -> 1), Swish activations, linear output.
    Matches the paper's "two hidden layers of 50 neurons, swish, linear output".
    """
    def __init__(self, width: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            Swish(),
            nn.Linear(width, width),
            Swish(),
            nn.Linear(width, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u shape: (B,1) or (1,1)
        return self.net(u)


class RMatrixNet(nn.Module):
    """
    Parameterizes R(u) ∈ C^{(d^2)×(d^2)} using real MLPs for each entry:
      R_ij(u) = a_ij(u) + i b_ij(u)
    """
    def __init__(self, d: int = 2, width: int = 50, device=None):
        super().__init__()
        self.d = d
        self.N = d * d  # R is N×N
        self.real_mlps = nn.ModuleList([ScalarMLP(width) for _ in range(self.N * self.N)])
        self.imag_mlps = nn.ModuleList([ScalarMLP(width) for _ in range(self.N * self.N)])
        self.to(device=device)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Returns R(u) as complex tensor of shape (N, N) if u is scalar-like (1,1),
        or (B, N, N) if u is batch (B,1).
        """
        if u.dim() != 2 or u.size(-1) != 1:
            raise ValueError(f"Expected u shape (B,1) or (1,1), got {tuple(u.shape)}")

        B = u.size(0)
        # Evaluate all entries; store in (B, N*N)
        re_vals = []
        im_vals = []
        for k in range(self.N * self.N):
            re_vals.append(self.real_mlps[k](u))  # (B,1)
            im_vals.append(self.imag_mlps[k](u))  # (B,1)
        re = torch.cat(re_vals, dim=1)  # (B, N*N)
        im = torch.cat(im_vals, dim=1)  # (B, N*N)

        R = torch.complex(re, im).view(B, self.N, self.N)  # (B,N,N)

        if B == 1:
            return R[0]  # (N,N)
        return R

    @torch.no_grad()
    def sanity_check_shapes(self):
        u = torch.zeros((3, 1), device=next(self.parameters()).device)
        R = self.forward(u)
        assert R.shape == (3, self.N, self.N)

    def dR_du_at0(self) -> torch.Tensor:
        """
        Compute derivative dR/du at u=0 via autograd, returning complex (N,N).
        Since we parameterize via a_ij and b_ij separately, we differentiate them separately.
        """
        device = next(self.parameters()).device
        u0 = torch.zeros((1, 1), device=device, dtype=torch.float32, requires_grad=True)

        # Build re/im matrices entrywise and differentiate each scalar output wrt u0.
        dRe = torch.zeros((self.N, self.N), device=device, dtype=torch.float32)
        dIm = torch.zeros((self.N, self.N), device=device, dtype=torch.float32)

        for i in range(self.N):
            for j in range(self.N):
                k = i * self.N + j
                a = self.real_mlps[k](u0)  # (1,1)
                b = self.imag_mlps[k](u0)  # (1,1)

                da = torch.autograd.grad(a, u0, retain_graph=True, create_graph=False)[0]  # (1,1)
                db = torch.autograd.grad(b, u0, retain_graph=True, create_graph=False)[0]  # (1,1)

                dRe[i, j] = da.squeeze()
                dIm[i, j] = db.squeeze()

        return torch.complex(dRe, dIm)


# -------------------------
# Losses
# -------------------------

def frob_sq(M: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm (sum |M_ij|^2)."""
    return (M.abs() ** 2).sum()


def ybe_loss_batch(
    model: RMatrixNet,
    u: torch.Tensor,
    v: torch.Tensor,
    I: torch.Tensor,
    P23: torch.Tensor,
) -> torch.Tensor:
    """
    YBE loss over a batch:
      || R12(u-v) R13(u) R23(v) - R23(v) R13(u) R12(u-v) ||_F^2
    We average over batch.

    u, v: shape (B,1), real.
    """
    device = next(model.parameters()).device
    B = u.size(0)
    loss = torch.zeros((), device=device, dtype=torch.float32)

    # Model outputs batched R matrices (B,N,N)
    Ru = model(u)         # (B,N,N)
    Rv = model(v)         # (B,N,N)
    Ruv = model(u - v)    # (B,N,N)

    for b in range(B):
        R_u = Ru[b]
        R_v = Rv[b]
        R_uv = Ruv[b]

        R12_uv = embed_R12(R_uv, I)
        R23_v = embed_R23(R_v, I)
        R13_u = embed_R13(R_u, I, P23)

        lhs = R12_uv @ R13_u @ R23_v
        rhs = R23_v @ R13_u @ R12_uv

        resid = lhs - rhs
        loss = loss + frob_sq(resid).real  # ensure float

    return loss / B


def regularity_loss(model: RMatrixNet, P: torch.Tensor) -> torch.Tensor:
    """||R(0) - P||_F^2"""
    device = next(model.parameters()).device
    u0 = torch.zeros((1, 1), device=device, dtype=torch.float32)
    R0 = model(u0)  # (N,N), complex
    return frob_sq(R0 - P.to(device=device)).real


def hamiltonian_loss(model: RMatrixNet, P: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    || P * R'(0) - H ||_F^2
    H should be (N,N) complex or real tensor on the same device (we cast to complex).
    """
    device = next(model.parameters()).device
    dR0 = model.dR_du_at0()  # (N,N), complex
    P = P.to(device=device)
    Hc = H.to(device=device)
    if not torch.is_complex(Hc):
        Hc = torch.complex(Hc, torch.zeros_like(Hc))
    resid = (P.to(dtype=torch.complex64) @ dR0.to(dtype=torch.complex64)) - Hc.to(dtype=torch.complex64)
    return frob_sq(resid).real


# -------------------------
# Training config & loop
# -------------------------

@dataclass
class TrainConfig:
    d: int = 2
    width: int = 50
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-3
    umin: float = -1.0
    umax: float = 1.0
    w_reg: float = 1.0
    w_H: float = 0.0  # set >0 if you pass H
    print_every: int = 10
    seed: int = 0
    device: str = "cpu"


def sample_uv(batch_size: int, umin: float, umax: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    u = (umax - umin) * torch.rand((batch_size, 1), device=device) + umin
    v = (umax - umin) * torch.rand((batch_size, 1), device=device) + umin
    return u, v


def train_solver(config: TrainConfig, H: Optional[torch.Tensor] = None):
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    model = RMatrixNet(d=config.d, width=config.width, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))

    # Fixed operators
    I = torch.eye(config.d, device=device, dtype=torch.float32)
    P = permutation_swap_VV(config.d, device=device, dtype=torch.float32)              # (d^2,d^2)
    P23 = permutation_swap_23_VVV(config.d, device=device, dtype=torch.float32)        # (d^3,d^3)

    for epoch in range(1, config.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        u, v = sample_uv(config.batch_size, config.umin, config.umax, device=str(device))

        L_ybe = ybe_loss_batch(model, u, v, I, P23)
        L_reg = regularity_loss(model, P) if config.w_reg != 0 else torch.zeros_like(L_ybe)

        if config.w_H != 0:
            if H is None:
                raise ValueError("config.w_H != 0 but no Hamiltonian H was provided.")
            L_H = hamiltonian_loss(model, P, H)
        else:
            L_H = torch.zeros_like(L_ybe)

        loss = L_ybe + config.w_reg * L_reg + config.w_H * L_H
        loss.backward()
        opt.step()

        if epoch % config.print_every == 0 or epoch == 1:
            print(
                f"epoch={epoch:4d}  "
                f"L_total={loss.item():.4e}  "
                f"L_ybe={L_ybe.item():.4e}  "
                f"L_reg={L_reg.item():.4e}  "
                f"L_H={L_H.item():.4e}"
            )

    return model


if __name__ == "__main__":
    cfg = TrainConfig(
        d=2,
        width=50,
        batch_size=16,
        epochs=200,
        lr=1e-3,
        w_reg=1.0,
        w_H=0.0,          # set to >0 and pass H to enforce P R'(0) = H
        device="cuda" if torch.cuda.is_available() else "cpu",
        print_every=10,
        seed=0,
    )

    # Example: no Hamiltonian constraint (just YBE + regularity)
    model = train_solver(cfg, H=None)

    # To add a Hamiltonian constraint:
    # N = cfg.d * cfg.d
    # H = torch.zeros((N, N), dtype=torch.float32)   # <-- replace with your target two-site H
    # cfg.w_H = 1.0
    # model = train_solver(cfg, H=H)