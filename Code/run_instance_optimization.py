import torch
import torch.nn.functional as F


def refine_displacement(
    fixed,
    moving,
    disp0,
    num_iter=300,
    lr=1e-2,
    lambda_reg=0.5,
    align_corners=True,
):
    """
    Minimal ConvexAdam-style instance optimization.

    fixed, moving : (1,1,D,H,W)
    disp0          : (1,3,D,H,W) voxel units
    returns refined disp (same shape)
    """

    device = fixed.device
    _, _, D, H, W = fixed.shape

    # --- Identity grid in normalized coords [-1,1]
    zs = torch.linspace(-1, 1, D, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)

    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid0 = torch.stack((xx, yy, zz), dim=-1)[None]  # (1,D,H,W,3)

    # --- Optimize displacement
    disp = disp0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([disp], lr=lr)

    # --- voxel → normalized scaling
    if align_corners:
        sx = 2.0 / (W - 1)
        sy = 2.0 / (H - 1)
        sz = 2.0 / (D - 1)
    else:
        sx = 2.0 / W
        sy = 2.0 / H
        sz = 2.0 / D

    for _ in range(num_iter):

        # --- Convert voxel disp → normalized grid offset
        disp_norm = torch.stack(
            (
                disp[:, 0] * sx,
                disp[:, 1] * sy,
                disp[:, 2] * sz,
            ),
            dim=-1,
        )  # (1,D,H,W,3)

        grid = grid0 + disp_norm

        warped = F.grid_sample(
            moving,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        )

        # --- SSD similarity
        loss_sim = ((fixed - warped) ** 2).mean()

        # --- Diffusion regularization
        dx = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
        dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
        dz = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]

        loss_reg = dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()

        loss = loss_sim + lambda_reg * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return disp.detach()

