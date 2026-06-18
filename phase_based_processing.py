import numpy as np
import cv2
import torch
import torch.nn.functional as F
from self_pyramid_utils import build_level, \
                         recon_level


class PhaseBased():

    def _init_(self,
                 sigma,
                 transfer_function,
                 phase_mag,
                 attenuate,
                 ref_idx,
                 batch_size,
                 device,
                 mag_mode="global",
                 spatial_chunks=1,
                 eps=1e-6):
        """
            sigma - Controls the amount of Amplitude-Weighted Phase Blurring
            transfer_function - Frequency Domain Bandpass Filter
            phase_mag - Phase Magnification factor
            attenuate - Determines whether to attenuate other frequencies
            ref_idx - Index of reference frame to compare local phase
            batch_size - Determines how many levels to process at the same time
            device - "cuda" or "cpu", tells PyTorch where to perform the calculations
            mag_mode - Chooses between "global" and "local" adaptive magnification
            spatial_chunks - Splits video frames into horizontal strips to save GPU memory during filtering
            eps - offset to avoid division by zero
        """
        self.sigma = sigma
        self.transfer_function = transfer_function
        self.phase_mag = phase_mag
        self.attenuate = attenuate
        self.ref_idx = ref_idx
        self.batch_size = batch_size
        self.device = device
        self.mag_mode = mag_mode
        self.spatial_chunks = spatial_chunks
        self.eps = eps
        self.gauss_kernel = self.get_gauss_kernel()


    def get_gauss_kernel(self):
        """ Obtains Gaussian Kernel for Aplitude weighted Blurring """
        ksize = np.max((3, np.ceil(4*self.sigma) - 1)).astype(int)
        if ((ksize % 2) != 1):
            ksize += 1
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=self.sigma)
        gauss_kernel = torch.tensor(gk @ gk.T).type(torch.float32) \
                                             .to(self.device) \
                                             .unsqueeze(0) \
                                             .unsqueeze(0)
        return gauss_kernel

    def adaptive_magnification_global(self, delta, base_mag, min_mag=0.5, max_mag=6.0):
        """ (Global) Reduces magnification based on the frame's average motion """
        motion_strength = torch.mean(torch.abs(delta))
        norm_strength = torch.clamp(motion_strength / 0.2, 0, 1)
        adaptive_mag = base_mag * (1 - 0.7 * norm_strength)
        adaptive_mag = torch.clamp(adaptive_mag, min_mag, max_mag)
        return adaptive_mag

    def adaptive_magnification_local(self, delta, base_mag, min_mag=0.5, max_mag=6.0):
        """ (Local) Reduces magnification on a per-pixel basis in high-motion areas. """
        motion_strength_map = torch.abs(delta)
        norm_strength_map = torch.clamp(motion_strength_map / 0.2, 0, 1)
        adaptive_mag_map = base_mag * (1 - 0.7 * norm_strength_map)
        adaptive_mag_map = torch.clamp(adaptive_mag_map, min_mag, max_mag)
        return adaptive_mag_map


    def process_single_channel(self,
                               frames_tensor,
                               filters_tensor,
                               video_dft):
        """ Applies Phase Based Processing in the Frequency Domain """

        num_frames, h, w = frames_tensor.shape[:3]
        num_filters, _, _ = filters_tensor.shape

        recon_dft = torch.zeros((num_frames, h, w),
                                dtype=torch.complex64).to(self.device)
        phase_deltas = torch.zeros((self.batch_size, num_frames, h, w),
                                    dtype=torch.complex64).to(self.device)

        for level in range(1, num_filters - 1, self.batch_size):
            print(f"[INFO] Processing levels {level}-{level+self.batch_size-1} with FFT (Slow, High Memory)...")
            idx1 = level
            idx2 = level + self.batch_size
            filter_batch = filters_tensor[idx1:idx2]

            ref_pyr = build_level(
                video_dft[self.ref_idx, :, :].unsqueeze(0), filter_batch)
            ref_phase = torch.angle(ref_pyr)

            for vid_idx in range(num_frames):
                curr_pyr = build_level(
                    video_dft[vid_idx, :, :].unsqueeze(0), filter_batch)
                _delta = torch.angle(curr_pyr) - ref_phase
                phase_deltas[:, vid_idx, :, :] = ((torch.pi + _delta) \
                                                  % 2*torch.pi) - torch.pi

            ## Temporally Filter the phase deltas

            # --- Spatial chunking (Memory-saving path) ---
            if self.spatial_chunks > 1:
                print(f"[INFO] ...using {self.spatial_chunks} spatial chunks.")
                filtered_deltas = torch.zeros((self.batch_size, num_frames, h, w), dtype=torch.float32).to(self.device)
                chunk_size = int(np.ceil(h / self.spatial_chunks))
                for i in range(self.spatial_chunks):
                    h1, h2 = i * chunk_size, min((i + 1) * chunk_size, h)
                    if h1 >= h2: break
                    delta_chunk = phase_deltas[:, :, h1:h2, :]
                    filtered_chunk = torch.fft.ifft(self.transfer_function * torch.fft.fft(delta_chunk, dim=1), dim=1).real
                    filtered_deltas[:, :, h1:h2, :] = filtered_chunk
                    del delta_chunk, filtered_chunk
                    if self.device == 'cuda': torch.cuda.empty_cache()
                phase_deltas = filtered_deltas
                del filtered_deltas
                if self.device == 'cuda': torch.cuda.empty_cache()
            else:
                # --- High-memory path (process all at once) ---
                print("[INFO] ...processing all frames at once (may crash).")
                phase_deltas = torch.fft.ifft(self.transfer_function \
                                            * torch.fft.fft(phase_deltas, dim=1),
                                            dim=1).real

            ## Apply Motion Magnifications
            for vid_idx in range(num_frames):
                vid_dft = video_dft[vid_idx, :, :].unsqueeze(0)
                curr_pyr = build_level(vid_dft, filter_batch)
                delta = phase_deltas[:, vid_idx, :, :].unsqueeze(1)

                # --- Amplitude-Weighted Blurring ---
                if self.sigma != 0:
                    amplitude_weight = (torch.abs(curr_pyr) + self.eps).unsqueeze(1)
                    weight = F.conv2d(input=amplitude_weight, weight=self.gauss_kernel, padding='same').squeeze(1)
                    delta = F.conv2d(input=(amplitude_weight * delta), weight=self.gauss_kernel, padding='same').squeeze(1)
                    delta /= (weight + self.eps)
                else:
                    delta = delta.squeeze(1)

                # --- Adaptive Magnification ---
                if self.mag_mode == "global":
                    adaptive_mag = self.adaptive_magnification_global(delta, self.phase_mag)
                    modifed_phase = delta * adaptive_mag
                else:
                    adaptive_mag_map = self.adaptive_magnification_local(delta, self.phase_mag)
                    modifed_phase = delta * adaptive_mag_map

                # --- Amplitude Attenuation ---
                if self.attenuate:
                    curr_pyr = torch.abs(curr_pyr) * (ref_pyr/torch.abs(ref_pyr))

                # --- Apply Amplified Phase ---
                curr_pyr = curr_pyr * torch.exp(1.0j*modifed_phase)
                recon_dft[vid_idx, :, :] += recon_level(curr_pyr, filter_batch).sum(dim=0)

            # --- Free memory after each level ---
            del phase_deltas
            if self.device == 'cuda': torch.cuda.empty_cache()
            phase_deltas = torch.zeros((self.batch_size, num_frames, h, w),
                                        dtype=torch.complex64).to(self.device)


        ## Add unchanged Low Pass Component for contrast
        lopass = filters_tensor[-1]
        for vid_idx in range(num_frames):
            curr_pyr_lo = build_level(video_dft[vid_idx, :, :], lopass)
            dft_lo = torch.fft.fftshift(torch.fft.fft2(curr_pyr_lo))
            recon_dft[vid_idx, :, :] += dft_lo*lopass


        ## GPU VRAM Optimization: Reconstruct one frame at a time
        print("[INFO] Reconstructing video from frequency domain (frame by frame)...")
        result_video = torch.zeros((num_frames, h, w), dtype=torch.float32).to(self.device)
        for vid_idx in range(num_frames):
            if (vid_idx + 1) % 100 == 0:
                print(f"[INFO] ...reconstructing frame {vid_idx + 1}/{num_frames}")
            frame_dft = recon_dft[vid_idx, :, :]
            frame_recon = torch.fft.ifft2(torch.fft.ifftshift(frame_dft)).real
            result_video[vid_idx, :, :] = frame_recon

        print("[INFO] Reconstruction complete.")
        del recon_dft
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return result_video