# Last modified: 2025-10-17

from typing import List, Union

import matplotlib
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from PIL import Image
import logging


def generate_gif_from_images(
    images_ls: List[Union[Image.Image, np.ndarray]],
    output_path: str,
    duration=500,
    loop=0,
    optimize=False,
):
    """
    Generate a GIF from multiple images.

    duration (`int`): the amount of time (in milliseconds) that each frame of the GIF will be displayed
    loop (`int`):
        0: The GIF will loop indefinitely.
        1: The GIF will play once and then stop.
        n: The GIF will play n times and then stop.
    """
    pil_image_ls = []
    for img in images_ls:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        pil_image_ls.append(img)

    # Save images as an animated GIF
    pil_image_ls[0].save(
        output_path,
        save_all=True,
        append_images=pil_image_ls[1:],
        format="GIF",
        duration=duration,
        loop=loop,
        optimize=optimize,
    )


def generate_video_from_image_paths(
    image_paths: List[str],
    output_path: str,
    fps: int = 30,
    bitrate: int = 1000,
    verbose=True,
):
    """
    Create a video from a list of image file paths using MoviePy.

    :param image_paths: List of paths to image files
    :param output_path: Path where the output video will be saved
    :param fps: Frames per second for the output video
    :param bitrate: Bitrate for the output video
    """
    if not image_paths:
        raise ValueError("The list of image paths is empty.")

    # Create a clip from the image sequence
    clip = ImageSequenceClip(image_paths, fps=fps)

    # Set the duration explicitly
    clip = clip.set_duration(len(image_paths) / fps)

    # 保证高是偶数，避免 yuv420p 编码出错
    clip = clip.crop(x1=0, y1=0, x2=clip.w, y2=clip.h - (clip.h % 2))

    # Write the clip to a video file
    clip.write_videofile(
        output_path,
        codec="libx264",
        bitrate=f"{bitrate}K",
        # ffmpeg_params=["-profile:v", "high", "-pix_fmt", "yuv420p"],
        verbose=verbose,
        logger=None if not verbose else "bar",
    )

    # logging.info(f"Video created successfully: {output_path}")


def generate_video_from_images(images, output_path, fps=24, bitrate=1000, verbose=True):
    """
    Generate a video from multiple images using moviepy.

    :param images: List of PIL images or NumPy arrays.
    :param output_path: Path to save the output video file.
    :param fps: Frames per second for the video (default is 24).
    :param bitrate: Bitrate for the video encoding (default is 1000).
    """
    # Convert PIL images to NumPy arrays if necessary
    images = [np.array(img) if isinstance(img, Image.Image) else img for img in images]

    # Fix dimensions in-place
    for i, image in enumerate(images):
        height, width = image.shape[:2]

        # Handle odd height
        if height % 2 != 0:
            image = image[1:]

        # Handle odd width
        if width % 2 != 0:
            image = image[:, 1:]

        images[i] = image

    # Create a video clip from the image sequence
    clip = ImageSequenceClip(images, fps=fps)

    # Set the duration explicitly
    clip = clip.set_duration(len(images) / fps)

    try:
        # Write the video file with the specified bitrate
        clip.write_videofile(
            output_path,
            codec="libx264",
            bitrate=f"{bitrate}k",
            fps=fps,
            verbose=verbose,
            logger=None if not verbose else "bar",
            preset="medium",  # Balanced preset for compatibility
            audio=False,  # No audio needed for image sequence
            threads=4,  # Limit threads to prevent memory issues
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",  # Ensure compatibility with QuickTime
                "-movflags",
                "+faststart",  # Enable streaming optimization
            ],
        )
    finally:
        clip.close()

    return output_path


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="plasma_r", valid_mask=None
):
    """

    Args:
        depth_map:
        cmap: matplotlib color map
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def combine_images_side_by_side(image1, image2, middle_margin=0):
    # Get the sizes of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with the combined width and the maximum height, including margin
    combined_width = width1 + width2 + middle_margin
    combined_height = max(height1, height2)

    combined_image = Image.new("RGB", (combined_width, combined_height))

    # Paste the images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1 + middle_margin, 0))

    return combined_image
