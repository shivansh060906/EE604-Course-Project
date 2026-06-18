# MotionMag — Phase-Based Video Motion Magnification

Phase-based motion magnification to reveal invisible movements (heartbeats, breathing, vibrations) in video.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Deploying to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Select your repo, branch `main`, and set Main file path to `app.py`.
4. Click **Deploy**.

Your app gets a public URL like `https://your-name-motionmag-app.streamlit.app`.
Free, HTTPS, auto-restarts.

> CPU only on the free tier — a 10-second 480p video takes ~1–3 minutes.

---

## Hugging Face Spaces (free GPU)

1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** as the SDK, hardware = **T4 small**.
3. Push all files to the Space repo.

---

## File structure

```
MotionMag/
├── app.py                     ← Streamlit UI
├── run_magnification.py       ← CLI / processing entry point
├── phase_based_processing.py
├── steerable_pyramid.py
├── self_phase_utils.py
├── self_pyarmid_utils.py
├── requirements.txt
└── README.md
```

## Recommended settings

| Use case             | Freq Lo | Freq Hi | α  |
|----------------------|---------|---------|----|
| Heartbeat (resting)  | 0.83    | 1.0     | 50 |
| Breathing            | 0.1     | 0.5     | 30 |
| Structural vibration | 0.5     | 10.0    | 25 |
