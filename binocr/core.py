from __future__ import annotations

import re
from typing import List

import numpy as np
import pytesseract
import cv2


def preprocess_for_01(image_bgr: np.ndarray, invert: bool = False, debug: bool = False) -> np.ndarray:
	"""Return a high-contrast, strictly black-white image emphasizing 0/1 glyphs.

	Pipeline:
	- Convert to grayscale
	- Optional inversion
	- Contrast equalization (CLAHE)
	- Strong sharpening (unsharp mask)
	- Otsu binary threshold (pure 0/255)
	- Light morphological opening to remove speckles
	"""
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	if invert:
		gray = cv2.bitwise_not(gray)

	# Contrast equalization to normalize lighting
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	eq = clahe.apply(gray)

	# Unsharp masking: sharpen edges aggressively
	blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=1.2, sigmaY=1.2)
	sharp = cv2.addWeighted(eq, 1.8, blur, -0.8, 0)

	# Otsu threshold to get crisp black-white
	_, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# Clean small artifacts
	kernel = np.ones((2, 2), np.uint8)
	opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

	if debug:
		cv2.imwrite("_debug_gray.png", gray)
		cv2.imwrite("_debug_eq.png", eq)
		cv2.imwrite("_debug_sharp.png", sharp)
		cv2.imwrite("_debug_bw.png", bw)
		cv2.imwrite("_debug_opened.png", opened)

	return opened


def ocr_array_to_bits(image_bgr: np.ndarray, invert: bool = False) -> str:
	"""OCR from an image array and return a string of 0/1 only."""
	proc = preprocess_for_01(image_bgr, invert=invert, debug=False)
	custom_oem_psm_config = "--oem 3 --psm 6 -c tessedit_char_whitelist=01"
	text = pytesseract.image_to_string(proc, config=custom_oem_psm_config)
	filtered = re.sub(r"[^01]", "", text)
	return filtered


def bits_to_text(bits: str, bits_per_char: int = 8, encoding: str = "utf-8") -> str:
	"""Convert a sequence of bits to text."""
	only_bits = re.sub(r"[^01]", "", bits)
	if bits_per_char <= 0:
		raise ValueError("bits_per_char must be positive")
	usable_len = (len(only_bits) // bits_per_char) * bits_per_char
	only_bits = only_bits[:usable_len]
	if usable_len == 0:
		return ""
	byte_values: List[int] = []
	for i in range(0, usable_len, bits_per_char):
		chunk = only_bits[i : i + bits_per_char]
		byte_values.append(int(chunk, 2))
	data = bytes(byte_values)
	return data.decode(encoding, errors="replace")

	
def _wrap_text_lines(text: str, max_width: int, font, font_scale: float, thickness: int) -> List[str]:
	words = text.split()
	lines: List[str] = []
	current: List[str] = []
	for word in words:
		candidate = (" ".join(current + [word])).strip()
		size, _ = cv2.getTextSize(candidate, font, font_scale, thickness)
		if size[0] <= max_width or not current:
			current.append(word)
		else:
			lines.append(" ".join(current))
			current = [word]
	if current:
		lines.append(" ".join(current))
	return lines


def render_side_panel(frame_bgr: np.ndarray, text: str, panel_width: int = 320, alpha: float = 0.85, font_scale: float = 0.7, thickness: int = 2, header: str = "Decoded") -> np.ndarray:
	"""Render a right-side panel with decoded text, keeping the live frame intact on the left."""
	h, w = frame_bgr.shape[:2]
	panel_w = max(200, min(panel_width, w))
	panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
	panel[:, :] = (0, 0, 0)
	cv2.addWeighted(panel, alpha, panel, 0, 0, dst=panel)  # keep black but allow alpha param use
	# Header
	font = cv2.FONT_HERSHEY_SIMPLEX
	header_text = header
	head_size, _ = cv2.getTextSize(header_text, font, font_scale + 0.2, thickness + 1)
	cv2.putText(panel, header_text, (12, 16 + head_size[1]), font, font_scale + 0.2, (0, 200, 255), thickness + 1, cv2.LINE_AA)
	# Body lines
	max_text_width = panel_w - 24
	lines = _wrap_text_lines(text if text.strip() else "(no data)", max_text_width, font, font_scale, thickness)
	line_height = int(1.6 * cv2.getTextSize("Ag", font, font_scale, thickness)[0][1])
	base_y = 16 + head_size[1] + 12
	for i, line in enumerate(lines):
		y = base_y + (i + 1) * line_height
		if y + 8 > h:
			break
		cv2.putText(panel, line, (12, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
	# Stack frame and panel
	stacked = np.concatenate([frame_bgr, panel], axis=1)
	return stacked
