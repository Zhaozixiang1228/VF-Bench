# Last modified: 2025-03-24

import torch
import torchvision
import torch.nn.functional as F
import einops
import numpy as np


def adaptive_gaussian_blur(
    image: torch.Tensor,
    radius_map: torch.Tensor,
    radius_sigma_ratio: int = 3,
    device=None,
):
    if device is None:
        device = image.device

    image = image.to(device)
    C, H, W = image.shape

    # Round and make sure radius_map is positive
    radius_map = torch.round(radius_map).int()
    radius_map = torch.clamp(radius_map, min=0).to(device)

    # Get unique radii values
    unique_radi = torch.unique(radius_map)

    # Initialize output tensor
    output = torch.zeros_like(image).to(device)

    # Process each unique radius
    for radius in unique_radi:
        radius = radius.item()

        # Create mask for current radius
        mask = (radius_map == radius).int().to(device)
        mask = einops.repeat(mask, "h w -> C h w", C=C)

        if 0 == radius:
            # no blur
            blurred = image.clone().to(device)

        else:
            # create Gaussian kernel
            kernel_size = 2 * radius + 1
            sigma = radius * 1.0 / radius_sigma_ratio
            ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
            x, y = torch.meshgrid(ax, ax, indexing="ij")
            gs_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
            gs_kernel /= gs_kernel.sum()  # Normalize to ensure sum = 1
            # reshape kernel
            gs_kernel = einops.repeat(gs_kernel, "h w -> C 1 h w", C=C)
            gs_kernel = gs_kernel.to(device)

            # Apply convolution with padding
            padding = int(kernel_size // 2)
            image_pad = F.pad(
                image, (padding, padding, padding, padding), mode="reflect"
            )
            blurred = F.conv2d(
                image_pad,
                gs_kernel,
                groups=C,
            )

        # Add to output using mask
        output += blurred * mask

        del blurred, mask

    return output


def blur_using_disp(
    image,  # [3 H W]
    disp_map,  # normalized disparity, [H W]
    focal_disp,  # normalized disparity
    blur_strength=0.1,
    max_blur_r=100,
    coc_smooth_kernel: int = 3,
    device=None,
):
    C, H, W = image.shape
    if device is None:
        device = image.device

    # Compute the Circle of Confusion (CoC) for each pixel.
    # # Reference: https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-23-depth-field-survey-techniques
    # coc_mm = np.abs(
    #     aperture_diam_mm
    #     * focal_length_mm
    #     * (depth_map - focal_depth)
    #     / (depth_map * (focal_depth - focal_length_mm))
    # )
    # Reference:  https://en.wikipedia.org/wiki/Circle_of_confusion#Determining_a_circle_of_confusion_diameter_from_the_object_field
    # coc_mm = (
    #     np.abs(depth_map - focal_depth)
    #     / depth_map
    #     * focal_length_mm**2
    #     / aperture_N
    #     / (focal_depth - focal_length_mm)
    # )

    # Approximation
    coc = np.abs(1 - disp_map / focal_disp) * focal_disp * blur_strength

    coc *= image.shape[-2]

    blur_radius = torch.from_numpy(coc)

    # Smooth the boundary (simulate feathering)
    blur_radius = torchvision.transforms.functional.gaussian_blur(
        blur_radius.unsqueeze(0), kernel_size=coc_smooth_kernel
    ).squeeze(0)

    blur_radius = torch.clamp(blur_radius, 0, max_blur_r).int()

    # Apply Gaussian with given CoC
    blurred_img = adaptive_gaussian_blur(
        image=torch.as_tensor(image),
        radius_map=blur_radius,
        device=device,
    )
    return blurred_img, blur_radius
