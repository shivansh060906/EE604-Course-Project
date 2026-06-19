import os, sys, re, datetime, argparse
import numpy as np
import cv2, torch

from steerable_pyramid import SteerablePyramid, SuboctaveSP
from phase_based_processing import PhaseBased
from self_phase_utils import *

EPS = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## 1. Setup Argument Parser
ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video_path", type=str, required=True)
ap.add_argument("-a", "--phase_mag", type=float, default=25.0)
ap.add_argument("-lo", "--freq_lo", type=float, required=True)
ap.add_argument("-hi", "--freq_hi", type=float, required=True)
ap.add_argument("-n", "--colorspace", type=str, default="luma3",
                choices={"luma1", "luma3", "gray", "yiq", "rgb"})
ap.add_argument("-p", "--pyramid_type", type=str, default="full_octave",
                choices={"full_octave", "half_octave",
                         "smooth_half_octave", "smooth_quarter_octave"})
ap.add_argument("-s", "--sigma", type=float, default=0.0)
ap.add_argument("-t", "--attenuate", type=bool, default=False)
ap.add_argument("-fs", "--sample_frequency", type=float, default=-1.0)
ap.add_argument("-r", "--reference_index", type=int, default=0)
ap.add_argument("-c", "--scale_factor", type=float, default=1.0)
ap.add_argument("-b", "--batch_size", type=int, default=2)
ap.add_argument("-d", "--save_directory", type=str, default="")
ap.add_argument("-mm", "--mag_mode", type=str, default="global",
                choices={"global", "local"})
ap.add_argument("-sc", "--spatial_chunks", type=int, default=1,
                help="Number of spatial chunks to process for saving memory. (e.g., 10)")

if __name__ == '__main__':

    print(f"[INFO] Script started. Using device: {DEVICE}")
    args = vars(ap.parse_args())

    print("[INFO] Running with the following arguments:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    ## 2. Setup File Paths
    video_path = args["video_path"]
    print(f"[INFO] Input video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit()

    save_dir = args["save_directory"] or os.path.dirname(video_path)
    if not os.path.exists(save_dir):
        print(f"Save directory not found, using input video directory")
        save_dir = os.path.dirname(video_path)
    print(f"[INFO] Save directory set to: {save_dir}")

    video_name = re.search(r"\w+(?=\.\w+)", video_path).group()

    # --- Generate a descriptive output filename ---
    video_save = os.path.join(save_dir,
                              f"{video_name}{args['colorspace']}{args['mag_mode']}fft{int(args['phase_mag'])}x.mp4")
    print(f"[INFO] Output video will be saved as: {video_save}")

    # --- Colorspace setup ---
    print(f"[INFO] Setting up for colorspace: {args['colorspace']}")
    if args["colorspace"] == "luma1":
        colorspace_func = lambda x: bgr2yiq(x)[:, :, 0]
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
            cv2.COLOR_GRAY2BGR)
    elif args["colorspace"] in {"luma3", "yiq"}:
        colorspace_func = bgr2yiq
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3),
            cv2.COLOR_RGB2BGR)
    elif args["colorspace"] == "gray":
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
            cv2.COLOR_GRAY2BGR)
    else:  # rgb
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3),
            cv2.COLOR_RGB2BGR)

    ## 3. Load Video and Prepare Filters
    print("[INFO] Loading video...")
    frames, video_fs = get_video(video_path, args["scale_factor"], colorspace_func)
    if not frames:
        print("[ERROR] Could not load video frames. Check video path and file format.")
        sys.exit()
    ref = frames[args["reference_index"]]
    h, w = ref.shape[:2]
    num_frames = len(frames)
    print(f"[INFO] Video loaded. Frames: {num_frames}, Original FS: {video_fs}, Shape: ({h}, {w})")

    fs = args["sample_frequency"] if args["sample_frequency"] > 0 else video_fs
    print(f"[INFO] Processing with sample frequency: {fs} Hz")

    ## 4. Create the 1D temporal bandpass filter
    transfer = bandpass_filter(args["freq_lo"], args["freq_hi"], fs, num_frames, DEVICE)
    print(f"[INFO] (FFT) Bandpass filter created for {args['freq_lo']}Hz to {args['freq_hi']}Hz.")

    # --- Determine pyramid depth and type ---
    max_depth = int(np.floor(np.log2(min(h, w))) - 2)
    pyr_type = args["pyramid_type"]
    print(f"[INFO] Steerable pyramid type: {pyr_type}")

    # --- Initialize the steerable pyramid ---
    if pyr_type == "full_octave":
        csp = SteerablePyramid(max_depth, 4, 1, 1.0, True)
    elif pyr_type == "half_octave":
        csp = SteerablePyramid(max_depth, 8, 2, 0.75, True)
    elif pyr_type == "smooth_half_octave":
        csp = SuboctaveSP(max_depth, 8, 2, 6, True)
    else:
        csp = SuboctaveSP(max_depth, 8, 4, 6, True)

    # --- Get the 2D spatial pyramid filters ---
    filters, _ = csp.get_filters(h, w, cropped=False)
    filters = torch.tensor(np.array(filters), dtype=torch.float32).to(DEVICE)
    print(f"[INFO] Pyramid filters created and moved to {DEVICE}.")

    batch = args["batch_size"]
    if filters.shape[0] % batch != 0:
        print(
            f"[INFO] Initial batch size {batch} is not a divisor of filter count {filters.shape[0]}. Finding new batch size...")
        for b in range(batch, 0, -1):
            if filters.shape[0] % b == 0:
                batch = b
                break
    print(f"[INFO] Using batch size: {batch}")

    ## 5. Load Frames to GPU
    print(f"[INFO] Creating {num_frames}-frame tensor (this may take a moment)...")
    try:
        frames_t_cpu = torch.tensor(np.array(frames, dtype=np.float32))
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        print(f"[ERROR] CPU RAM Error: {e}")
        print("Your video is too large to load into RAM. Try resizing with -c 0.5")
        sys.exit()

    frames_t = frames_t_cpu.to(DEVICE)
    print(f"[INFO] Frames tensor moved to {DEVICE}.")
    del frames, frames_t_cpu  # Free CPU RAM

    ## 6. Initialize the Phase Processor
    pb = PhaseBased(args["sigma"],
                    transfer,
                    args["phase_mag"],
                    args["attenuate"],
                    args["reference_index"],
                    batch,
                    DEVICE,
                    mag_mode=args["mag_mode"],
                    spatial_chunks=args["spatial_chunks"],  # Pass the memory-fix argument
                    eps=EPS)

    print(f"[INFO] PhaseBased processor initialized (Mode: {args['mag_mode']}).")
    print("-" * 30)
    print(f"[INFO] Starting phase-based processing...")

    ## 7. Run Processing (Channel by Channel for Memory Safety)
    if args["colorspace"] in {"yiq", "rgb"}:
        print("[INFO] Processing all 3 color channels...")
        output = torch.zeros_like(frames_t)  # Final output tensor
        for c in range(frames_t.shape[-1]):
            print(f"[INFO] Processing channel {c + 1}/3...")
            channel_t = frames_t[:, :, :, c].clone()
            dft = get_fft2_batch(channel_t).to(DEVICE)
            output[:, :, :, c] = pb.process_single_channel(channel_t, filters, dft)
            del channel_t, dft
            if DEVICE == 'cuda': torch.cuda.empty_cache()
    elif args["colorspace"] == "luma3":
        print("[INFO] Processing luma channel (Y) only...")
        output = frames_t.clone()
        luma_t = frames_t[:, :, :, 0].clone()
        dft = get_fft2_batch(luma_t).to(DEVICE)
        del frames_t
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        output_luma = pb.process_single_channel(luma_t, filters, dft)
        output[:, :, :, 0] = output_luma
        del luma_t, dft, output_luma
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    else:  # gray or luma1
        print("[INFO] Processing single channel...")
        dft = get_fft2_batch(frames_t).to(DEVICE)
        output = pb.process_single_channel(frames_t, filters, dft)
        del dft, frames_t
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    print(f"[INFO] Processing complete. 'output' tensor is on {output.device}.")
    print(f"[INFO] Initializing video writer for saving...")
    out = cv2.VideoWriter(video_save, cv2.VideoWriter_fourcc(*'avc1'),
                          int(np.round(video_fs)),
                          (int(w / args["scale_factor"]), int(h / args["scale_factor"])))

    ## 8. Save Video (Frame by Frame for Memory Safety)
    print("[INFO] Writing frames one by one...")
    for i in range(num_frames):
        if (i + 1) % 50 == 0 or (i + 1) == num_frames:
            print(f"[INFO] Writing frame {i + 1}/{num_frames}...")
        frame_tensor = output[i]
        frame_numpy = frame_tensor.cpu().numpy()
        frame = inv_colorspace(frame_numpy)
        frame = cv2.resize(frame, (int(w / args["scale_factor"]), int(h / args["scale_factor"])))
        out.write(frame)

    print("[INFO] Video writer released.")
    out.release()
    del output
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    print(f"Saved: {video_save}")
    print("[INFO] Script finished.")