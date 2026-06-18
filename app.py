import streamlit as st
import os, re, sys, tempfile, subprocess
import numpy as np
import cv2
import torch

# --- Page config ---
st.set_page_config(
    page_title="MotionMag — Phase-Based Motion Magnification",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 MotionMag")
st.caption("Phase-based video motion magnification · EE604 Project")

st.markdown(
    "Upload a video and amplify invisible motions — heartbeats, breathing, "
    "structural vibrations — using steerable pyramid phase processing."
)

# --- Sidebar: parameters ---
with st.sidebar:
    st.header("⚙️ Parameters")

    phase_mag = st.slider("Magnification factor (α)", 1.0, 100.0, 25.0, 1.0,
                          help="How much to amplify the detected motion.")

    freq_lo = st.number_input("Low cutoff frequency (Hz)", 0.1, 30.0, 0.4, 0.1,
                               format="%.2f",
                               help="Lower bound of the temporal bandpass filter.")
    freq_hi = st.number_input("High cutoff frequency (Hz)", 0.1, 30.0, 3.0, 0.1,
                               format="%.2f",
                               help="Upper bound of the temporal bandpass filter.")

    colorspace = st.selectbox(
        "Colorspace",
        ["luma3", "luma1", "gray", "yiq", "rgb"],
        help="luma3 magnifies only the luminance channel (recommended)."
    )

    pyramid_type = st.selectbox(
        "Pyramid type",
        ["full_octave", "half_octave", "smooth_half_octave", "smooth_quarter_octave"],
    )

    mag_mode = st.selectbox("Magnification mode", ["global", "local"])

    sigma = st.slider("Amplitude-weighted blur (σ)", 0.0, 10.0, 0.0, 0.5,
                      help="Gaussian blur sigma for amplitude-weighted phase smoothing.")

    scale_factor = st.slider("Scale factor", 0.25, 1.0, 1.0, 0.25,
                              help="Resize video before processing to save memory.")

    spatial_chunks = st.slider("Spatial chunks", 1, 20, 1,
                                help="Split frames into horizontal strips to save GPU/CPU memory.")

    batch_size = st.slider("Batch size", 1, 16, 2,
                            help="Number of pyramid levels processed at once.")

    reference_index = st.number_input("Reference frame index", 0, 500, 0, 1)

    st.divider()
    device_label = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
    st.caption(f"Device: **{device_label}**")

# --- Main: upload & run ---
uploaded = st.file_uploader(
    "Upload a video", type=["mp4", "avi", "mov", "mkv"],
    help="Short clips (5–15 s) work best. Longer videos need more RAM/VRAM."
)

# --- Demo videos ---
st.markdown("---")
st.subheader("Demo Results")
st.caption("Original vs. magnified — side by side.")

DEMOS = [
    {
        "label": "Eye Twitching",
        "description": "Description of demo 1.",
        "original": "static/Eye.mp4",
        "amplified": "static/Eye.wmv",
    },
    {
        "label": "Patient Breathing",
        "description": "Description of demo 2.",
        "original": "static/Face.mp4",
        "amplified": "static/Face.wmv",
    },
    {
        "label": "Wrist Pulse",
        "description": "Description of demo 3.",
        "original": "static/Wrist.mp4",
        "amplified": "static/Wrist.wmv",
    },
]

for demo in DEMOS:
    st.markdown(f"#### {demo['label']}")
    col_orig, col_amp = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        if os.path.exists(demo["original"]):
            st.video(demo["original"])
        else:
            st.info("Video coming soon.")
    with col_amp:
        st.markdown("**Magnified**")
        if os.path.exists(demo["amplified"]):
            st.video(demo["amplified"])
        else:
            st.info("Video coming soon.")
    st.markdown("---")

if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded.name)
        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        # Show original
        st.subheader("Original video")
        st.video(input_path)

        if freq_lo >= freq_hi:
            st.error("Low cutoff must be less than high cutoff.")
            st.stop()

        if st.button("▶ Run Magnification", type="primary"):
            video_name = re.search(r"\w+(?=\.\w+)", uploaded.name).group()
            out_name = f"{video_name}_{colorspace}_{mag_mode}_fft{int(phase_mag)}x.mp4"
            output_path = os.path.join(tmpdir, out_name)

            cmd = [
                sys.executable, "run_magnification.py",
                "-v", input_path,
                "-a", str(phase_mag),
                "-lo", str(freq_lo),
                "-hi", str(freq_hi),
                "-n", colorspace,
                "-p", pyramid_type,
                "-s", str(sigma),
                "-mm", mag_mode,
                "-c", str(scale_factor),
                "-b", str(batch_size),
                "-r", str(reference_index),
                "-sc", str(spatial_chunks),
                "-d", tmpdir,
            ]

            log_box = st.empty()
            progress = st.progress(0, text="Starting…")

            with st.spinner("Processing… (this may take a while on CPU)"):
                proc = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                log_lines = []
                for line in proc.stdout:
                    log_lines.append(line.rstrip())
                    log_box.code("\n".join(log_lines[-20:]), language="")
                    # crude progress from log lines
                    if "Loading video" in line:
                        progress.progress(10, text="Loading video…")
                    elif "Bandpass filter" in line:
                        progress.progress(25, text="Creating filters…")
                    elif "Starting phase" in line:
                        progress.progress(40, text="Phase processing…")
                    elif "Writing frame" in line:
                        m = re.search(r"(\d+)/(\d+)", line)
                        if m:
                            pct = int(m.group(1)) / int(m.group(2))
                            progress.progress(int(40 + pct * 55), text=f"Writing frames… {m.group(1)}/{m.group(2)}")

                proc.wait()

            if proc.returncode != 0:
                st.error("Processing failed. Check the log above for details.")
            elif os.path.exists(output_path):
                progress.progress(100, text="Done!")
                st.success("✅ Magnification complete!")
                st.subheader("Magnified video")
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇ Download magnified video",
                        f,
                        file_name=out_name,
                        mime="video/mp4",
                    )
            else:
                # Find any mp4 in tmpdir
                candidates = [x for x in os.listdir(tmpdir) if x.endswith(".mp4") and x != uploaded.name]
                if candidates:
                    out_found = os.path.join(tmpdir, candidates[0])
                    progress.progress(100, text="Done!")
                    st.success("✅ Done!")
                    st.video(out_found)
                    with open(out_found, "rb") as f:
                        st.download_button("⬇ Download", f, file_name=candidates[0], mime="video/mp4")
                else:
                    st.error("Output file not found. Check the log.")