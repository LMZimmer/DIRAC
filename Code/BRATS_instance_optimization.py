import torch
import torch.nn.functional as F


# ---------- Warping utilities ----------

def make_identity_grid(D, H, W, device, dtype):
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)[None]  # (1,D,H,W,3)


def voxel_disp_to_norm(disp, D, H, W):
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    sz = 2.0 / max(D - 1, 1)
    return torch.stack((dx * sx, dy * sy, dz * sz), dim=-1)


def warp(img, disp):
    _, _, D, H, W = img.shape
    grid0 = make_identity_grid(D, H, W, img.device, img.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, D, H, W)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp_field(field, disp):
    _, _, D, H, W = field.shape
    grid0 = make_identity_grid(D, H, W, field.device, field.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, D, H, W)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ---------- NCC similarity ----------

def ncc_loss(I, J, win=3):
    pad = win // 2
    filt = torch.ones((1, 1, win, win, win), device=I.device)

    def conv(x):
        return F.conv3d(x, filt, padding=pad)

    I2, J2, IJ = I * I, J * J, I * J

    I_sum = conv(I)
    J_sum = conv(J)
    I2_sum = conv(I2)
    J2_sum = conv(J2)
    IJ_sum = conv(IJ)

    win_size = float(win ** 3)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    ncc = cross * cross / (I_var * J_var + 1e-5)
    return -ncc.mean()


# ---------- Regularization ----------

def smoothness(disp):
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    return dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()


# ---------- Inverse consistency ----------

def inv_consistency(d_fwd, d_bwd):
    bwd_warped = warp_field(d_bwd, d_fwd)
    fwd_warped = warp_field(d_fwd, d_bwd)
    return (d_fwd + bwd_warped).pow(2).mean() + (d_bwd + fwd_warped).pow(2).mean()


# ---------- Paper-accurate instance optimization ----------

def dirac_instance_optimization(
    B, F,
    disp_fb_init,
    disp_bf_init=None,
    lambdas_reg=[0.25, 0.3, 0.3, 0.35, 0.35],
    lambdas_inv=[1.0, 2.0, 4.0, 8.0, 10.0],
    lrs=[1e-2, 5e-3, 5e-3, 3e-3, 3e-3],
    iters=[150, 100, 100, 100, 50],
    lambda_m=0.01,
):
    """
    B: baseline (fixed) image (1,1,D,H,W)
    F: followup (moving) image (1,1,D,H,W)
    disp_fb_init: F->B field (1,3,D,H,W)
    disp_bf_init: B->F field (optional)
    """

    if disp_bf_init is None:
        disp_bf_init = -disp_fb_init

    # Masks initialized to zero (no exclusion)
    m_fb = torch.zeros_like(B)
    m_bf = torch.zeros_like(B)

    disp_fb = disp_fb_init.clone()
    disp_bf = disp_bf_init.clone()

    for lr, n_iter, lam_reg, lam_inv in zip(lrs, iters, lambdas_reg, lambdas_inv):

        disp_fb = disp_fb.detach().requires_grad_(True)
        disp_bf = disp_bf.detach().requires_grad_(True)
        m_fb = m_fb.detach().requires_grad_(True)
        m_bf = m_bf.detach().requires_grad_(True)

        opt = torch.optim.Adam([disp_fb, disp_bf, m_fb, m_bf], lr=lr)

        for _ in range(n_iter):

            F_warp = warp(F, disp_fb)
            B_warp = warp(B, disp_bf)

            # ---- Masked similarity Ls ----
            Ls = (
                ncc_loss(B * (1 - m_fb), F_warp * (1 - m_fb)) +
                ncc_loss(F * (1 - m_bf), B_warp * (1 - m_bf))
            )

            # ---- Regularization ----
            Lr = smoothness(disp_fb) + smoothness(disp_bf)

            # ---- Inverse consistency ----
            Linv = inv_consistency(disp_fb, disp_bf)

            # ---- Mask penalty ----
            Lm = m_fb.abs().mean() + m_bf.abs().mean()

            loss = (1 - lam_reg) * Ls + lam_reg * Lr + lam_inv * Linv + lambda_m * Lm

            opt.zero_grad()
            loss.backward()
            opt.step()

    return disp_fb.detach(), disp_bf.detach(), m_fb.detach(), m_bf.detach()

