## Live Binary OCR (Webcam → Text)

Minimal CLI to read 0/1 digits from a live camera and decode them as text.

### Demo

![Live demo](example.gif)

### Install

1) Python 3.10+
2) System Tesseract OCR installed and on PATH
   - Windows: download from `https://github.com/UB-Mannheim/tesseract/wiki`
   - Linux: `sudo apt install tesseract-ocr`
3) Python dependencies:
```bash
pip install -r requirements.txt
```

### Run (live camera)

```bash
python -m binocr live --camera-index 0 --bits-per-char 8 --encoding utf-8 --invert
```

Keys:
- Press `q` to quit the window.

Main options:
- `--camera-index`: OpenCV device index (default 0)
- `--bits-per-char`: bits per character (default 8)
- `--encoding`: output encoding (default utf-8)
- `--invert`: invert the image before OCR (useful for white-on-black)
- `--mirror`: flip the camera horizontally
- `--panel-width`: width of right-hand decoded text panel
- `--fps-limit`: OCR updates per second (default 8.0)

### Notes
- Keep the binary digits sharp and high-contrast for best OCR accuracy.
- Non 0/1 characters are filtered out before decoding.

### Troubleshooting
- Verify Tesseract is installed: `tesseract -v`
- Try `--invert` if foreground/background are reversed.

