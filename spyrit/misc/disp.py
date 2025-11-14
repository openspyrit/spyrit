# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import math
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Union


def display_vid(video, fps, title="", colormap=plt.cm.gray):
    """
    video is a numpy array of shape [nb_frames, 1, nx, ny]
    """
    plt.ion()
    (nb_frames, channels, nx, ny) = video.shape
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(nb_frames):
        current_frame = video[i, 0, :, :]
        plt.imshow(current_frame, cmap=colormap)
        plt.title(title)
        divider = make_axes_locatable(ax)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.show()
        plt.pause(fps)
    plt.ioff()


def display_rgb_vid(video, fps, title=""):
    """
    video is a numpy array of shape [nb_frames, 3, nx, ny]
    """
    plt.ion()
    (nb_frames, channels, nx, ny) = video.shape
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(nb_frames):
        current_frame = video[i, :, :, :]
        current_frame = np.moveaxis(current_frame, 0, -1)
        plt.imshow(current_frame)
        plt.title(title)
        plt.show()
        plt.pause(fps)
    plt.ioff()


def fitPlots(N, aspect=(16, 9)):
    width = aspect[0]
    height = aspect[1]
    area = width * height * 1.0
    factor = (N / area) ** (1 / 2.0)
    cols = math.floor(width * factor)
    rows = math.floor(height * factor)
    rowFirst = width < height
    while rows * cols < N:
        if rowFirst:
            rows += 1
        else:
            cols += 1
        rowFirst = not (rowFirst)
    return rows, cols


def Multi_plots(
    img_list,
    title_list,
    shape,
    suptitle="",
    colormap=plt.cm.gray,
    axis_off=True,
    aspect=(16, 9),
    savefig="",
    fontsize=14,
):
    [rows, cols] = shape
    plt.figure()
    plt.suptitle(suptitle, fontsize=16)
    if (len(img_list) < rows * cols) or (len(title_list) < rows * cols):
        for k in range(max(rows * cols - len(img_list), rows * cols - len(title_list))):
            img_list.append(np.zeros((64, 64)))
            title_list.append("")

    for k in range(rows * cols):
        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(img_list[k], cmap=colormap)
        ax.set_title(title_list[k], fontsize=fontsize)
        if axis_off:
            plt.axis("off")
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def compare_video_frames(
    vid_list,
    nb_disp_frames,
    title_list,
    suptitle="",
    colormap=plt.cm.gray,
    aspect=(16, 9),
    savefig="",
    fontsize=14,
):
    rows = len(vid_list)
    cols = nb_disp_frames
    plt.figure(figsize=aspect)
    plt.suptitle(suptitle, fontsize=16)
    for i in range(rows):
        for j in range(cols):
            k = (j + 1) + (i) * (cols)
            i
            # print(k)
            ax = plt.subplot(rows, cols, k)
            # print("i = {}, j = {}".format(i,j))
            ax.imshow(vid_list[i][0, j, 0, :, :], cmap=colormap)
            ax.set_title(title_list[i][j], fontsize=fontsize)
            plt.axis("off")
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def torch2numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy()


def uint8(dsp):
    x = (dsp - np.amin(dsp)) / (np.amax(dsp) - np.amin(dsp)) * 255
    x = x.astype("uint8")
    return x


def imagesc(
    Img,
    title="",
    colormap=plt.cm.gray,
    show=True,
    figsize=None,
    cbar_pos=None,
    title_fontsize=16,
):
    """
    imagesc(IMG) Display image Img with scaled colors with greyscale
    colormap and colorbar
    imagesc(IMG, title=ttl) Display image Img with scaled colors with
    greyscale colormap and colorbar, with the title ttl
    imagesc(IMG, title=ttl, colormap=cmap) Display image Img with scaled colors
    with colormap and colorbar specified by cmap (choose between 'plasma',
    'jet', and 'grey'), with the title ttl
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(Img, cmap=colormap)
    plt.title(title, fontsize=title_fontsize)
    divider = make_axes_locatable(ax)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if cbar_pos == "bottom":
        cax = inset_axes(
            ax, width="100%", height="5%", loc="lower center", borderpad=-5
        )
        plt.colorbar(cax=cax, orientation="horizontal")
    else:
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax, orientation="vertical")

    # fig.tight_layout() # it raises warnings in some cases
    if show is True:
        plt.show()


def imagecomp(
    Img1,
    Img2,
    suptitle="",
    title1="",
    title2="",
    colormap1=plt.cm.gray,
    colormap2=plt.cm.gray,
):
    f, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(Img1, cmap=colormap1)
    ax1.set_title(title1)
    cax = plt.axes([0.43, 0.3, 0.025, 0.4])
    plt.colorbar(im1, cax=cax)
    plt.suptitle(suptitle, fontsize=16)
    #
    im2 = ax2.imshow(Img2, cmap=colormap2)
    ax2.set_title(title2)
    cax = plt.axes([0.915, 0.3, 0.025, 0.4])
    plt.colorbar(im2, cax=cax)
    plt.subplots_adjust(left=0.08, wspace=0.5, top=0.9, right=0.9)
    plt.show()


def imagepanel(
    Img1,
    Img2,
    Img3,
    Img4,
    suptitle="",
    title1="",
    title2="",
    title3="",
    title4="",
    colormap1=plt.cm.gray,
    colormap2=plt.cm.gray,
    colormap3=plt.cm.gray,
    colormap4=plt.cm.gray,
):
    fig, axarr = plt.subplots(2, 2, figsize=(20, 10))
    plt.suptitle(suptitle, fontsize=16)

    im1 = axarr[0, 0].imshow(Img1, cmap=colormap1)
    axarr[0, 0].set_title(title1)
    cax = plt.axes([0.4, 0.54, 0.025, 0.35])
    plt.colorbar(im1, cax=cax)

    im2 = axarr[0, 1].imshow(Img2, cmap=colormap2)
    axarr[0, 1].set_title(title2)
    cax = plt.axes([0.90, 0.54, 0.025, 0.35])
    plt.colorbar(im2, cax=cax)

    im3 = axarr[1, 0].imshow(Img3, cmap=colormap3)
    axarr[1, 0].set_title(title3)
    cax = plt.axes([0.4, 0.12, 0.025, 0.35])
    plt.colorbar(im3, cax=cax)

    im4 = axarr[1, 1].imshow(Img4, cmap=colormap4)
    axarr[1, 1].set_title(title4)
    cax = plt.axes([0.9, 0.12, 0.025, 0.35])
    plt.colorbar(im4, cax=cax)

    plt.subplots_adjust(left=0.08, wspace=0.5, top=0.9, right=0.9)
    plt.show()


def plot(x, y, title="", xlabel="", ylabel="", color="black"):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def add_colorbar(mappable, position="right", size="5%"):
    """
    Example:
        f, axs = plt.subplots(1, 2)
        im = axs[0].imshow(img1, cmap='gray')
        add_colorbar(im)
        im = axs[0].imshow(img2, cmap='gray')
        add_colorbar(im)
    """
    if position == "bottom":
        orientation = "horizontal"
    else:
        orientation = "vertical"

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    plt.sca(last_axes)
    return cbar


def noaxis(axs):
    if type(axs) is np.ndarray:
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    else:
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)


def string_mean_std(x, prec=3):
    return "{:.{p}f} +/- {:.{p}f}".format(np.mean(x), np.std(x), p=prec)


def print_mean_std(x, tag="", prec=3):
    print("{} = {:.{p}f} +/- {:.{p}f}".format(tag, np.mean(x), np.std(x), p=prec))


def histogram(s):
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.show()


def vid2batch(root, img_dim, start_frame, end_frame):
    from imutils.video import FPS
    import imutils
    import cv2

    stream = cv2.VideoCapture(root)
    fps = FPS().start()
    frame_nb = 0
    output_batch = torch.zeros(1, end_frame - start_frame, 1, img_dim, img_dim)
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break

        frame_nb += 1
        if (frame_nb >= start_frame) & (frame_nb < end_frame):
            frame = cv2.resize(frame, (img_dim, img_dim))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            output_batch[0, frame_nb - start_frame, 0, :, :] = torch.Tensor(
                frame[:, :, 1]
            )

    return output_batch


def pre_process_video(video, crop_patch, kernel_size):
    import cv2

    batch_size, seq_length, c, h, w = video.shape
    batched_frames = video.reshape(batch_size * seq_length * c, h, w)
    output_batch = torch.zeros(batched_frames.shape)

    for i in range(batch_size * seq_length * c):
        img = torch2numpy(batched_frames[i, :, :])
        img[crop_patch] = 0
        median_frame = cv2.medianBlur(img, kernel_size)
        output_batch[i, :, :] = torch.Tensor(median_frame)
    output_batch = output_batch.reshape(batch_size, seq_length, c, h, w)
    return output_batch


def contrib_map(H_dyn: np.ndarray, n: int, save_figs: bool = False, 
                path_fig: Union[str, Path] = '', show_fig: bool = True) -> np.ndarray:
    """
    Generate a contribution map showing measurement pattern coverage.
    
    Args:
        H_dyn: Dynamic measurement matrix.
        n: Image size (assuming square images).
        save_figs: Whether to save the figure.
        path_fig: Path to save the figure.
        show_fig: Whether to display the figure.
        
    Returns:
        Contribution map as numpy array.
    """
    _, L = H_dyn.shape
    l = int(np.sqrt(L))
    
    if l * l != L:
        raise ValueError(f"Matrix dimension L={L} is not a perfect square")
    
    # Create contribution map
    contrib = np.where(H_dyn != 0, 1, 0)
    contrib_image = np.sum(contrib, axis=0)
    contrib_image = contrib_image.reshape((l, l)) / (n ** 2)
    contrib_image = np.rot90(contrib_image, 2)  # Account for rotation in SP POV
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(contrib_image, cmap="hot")
    cbar = plt.colorbar(fraction=0.047, pad=0.01, format="%.1f")
    cbar.ax.tick_params(labelsize=15)
    plt.title("Contribution Map", fontsize=16)
    
    if save_figs and path_fig:
        plt.axis('off')
        plt.savefig(str(path_fig), bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return contrib_image


def error_map(img1: np.ndarray, img2: np.ndarray, save_figs: bool = False, 
              path_fig: Union[str, Path] = '', show_fig: bool = True, 
              title: str = "Error Map") -> np.ndarray:
    """
    Compute and visualize error map between two images.
    
    Args:
        img1: First image.
        img2: Second image.
        save_figs: Whether to save the figure.
        path_fig: Path to save the figure.
        show_fig: Whether to display the figure.
        title: Title for the plot.
        
    Returns:
        Error map as numpy array.
        
    Raises:
        ValueError: If images have different shapes.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    # Normalize images to [0, 1]
    img1_normalized = (img1 - img1.min()) / (img1.max() - img1.min()) if img1.max() != img1.min() else np.zeros_like(img1)
    img2_normalized = (img2 - img2.min()) / (img2.max() - img2.min()) if img2.max() != img2.min() else np.zeros_like(img2)

    error = img1_normalized - img2_normalized
    
    plt.figure(figsize=(8, 6))
    plt.imshow(error, cmap='Spectral')
    cbar = plt.colorbar(fraction=0.047, pad=0.01, format="%.2f")
    cbar.ax.tick_params(labelsize=15)
    plt.title(title, fontsize=16)
    plt.axis('off')

    if save_figs and path_fig:
        plt.savefig(str(path_fig), bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close()
        
    return error


def blue_box(f: np.ndarray, amp_max: int = 0, box_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Add a colored box overlay to an image for visualization purposes.
    
    Args:
        f: Input grayscale or RGB image.
        amp_max: Offset from image borders for the box.
        box_color: RGB color for the box (default: blue).
        
    Returns:
        RGB image with colored box overlay.
        
    Raises:
        ValueError: If amp_max is too large for the image size.
    """
    if amp_max * 2 >= min(f.shape[0], f.shape[1]):
        raise ValueError(f"amp_max={amp_max} is too large for image shape {f.shape}")
    
    # Normalize to 8-bit and create RGB image
    f_normalized = (f - f.min()) / (f.max() - f.min()) if f.max() != f.min() else np.zeros_like(f)
    f_255 = (f_normalized * 255).astype(np.uint8)

    if len(f.shape) == 2:
        f_rgb = np.stack([f_255] * 3, axis=2)
    elif len(f.shape) == 3:
        f_rgb = f_255
    
    if amp_max == 0:
        return f_rgb
    
    r, g, b = box_color
    
    # Draw box borders
    # Top and bottom borders
    f_rgb[amp_max, amp_max:-amp_max] = [r, g, b]
    f_rgb[-amp_max-1, amp_max:-amp_max] = [r, g, b]
    
    # Left and right borders
    f_rgb[amp_max:-amp_max, amp_max] = [r, g, b]
    f_rgb[amp_max:-amp_max, -amp_max-1] = [r, g, b]

    return f_rgb



def get_frame(movie_path: Union[str, Path], frame_number: int = 0) -> np.ndarray:
    """
    Extract a specific frame from a video file.
    
    Args:
        movie_path: Path to the video file.
        frame_number: Frame number to extract (0-indexed).
        
    Returns:
        Grayscale frame as numpy array.
        
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If frame cannot be read or video is invalid.
    """
    movie_path = Path(movie_path)
    if not movie_path.exists():
        raise FileNotFoundError(f"Video file not found: {movie_path}")
    
    cap = cv2.VideoCapture(str(movie_path))

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {movie_path}")

    # Get total frame count for validation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_number} not available. Video has {total_frames} frames.")

    # Set frame position directly (more efficient than iterating)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from video")

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    return gray_frame


def save_motion_video(x_motion, out_path, amp_max=0, fps=820):
    r"""
    Save :attr:`x_motion` of shape (1, n_frames, c, h, w) as a video file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_batch, n_frames, n_wav, h, w = x_motion.shape

    h_crop, w_crop = torch.tensor((h, w)) - 2 * amp_max
    h_crop, w_crop = int(h_crop.item()), int(w_crop.item())

    if n_batch > 1:
        raise ValueError(f"save_motion_video expects a single batch, got {n_batch}")
    if n_wav != 1 and n_wav != 3:
        raise ValueError(f"save_motion_video expects 1 or 3 channels, got {n_wav}")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_crop, h_crop), True)
    if not writer.isOpened():
        raise RuntimeError("cv2 VideoWriter failed to open")
    for t in range(n_frames):
        frame_wide = x_motion[0, t].moveaxis(0, -1).cpu().numpy()

        mn, mx = frame_wide.min(), frame_wide.max()

        frame = frame_wide[amp_max:h - amp_max, amp_max:w - amp_max]

        if mx > mn:
            frame8 = ((frame - mn) / (mx - mn) * 255.0).astype('uint8')
        else:
            frame8 = (frame * 0).astype('uint8')
        # frame_bgr = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)
        frame_bgr = cv2.cvtColor(frame8, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"Saved motion video to {out_path}")