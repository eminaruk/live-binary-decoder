import click
from .core import ocr_array_to_bits, bits_to_text, render_side_panel
import cv2
import time


@click.group()
def cli() -> None:
	"""Binary OCR CLI."""
	pass


@cli.command()
@click.option("--camera-index", type=int, default=0, show_default=True, help="OpenCV camera index")
@click.option("--bits-per-char", type=int, default=8, show_default=True)
@click.option("--encoding", type=str, default="utf-8", show_default=True)
@click.option("--invert/--no-invert", default=False, show_default=True)
@click.option("--mirror/--no-mirror", default=False, show_default=True, help="Mirror (flip horizontally) the camera feed")
@click.option("--panel-width", type=int, default=360, show_default=True)
@click.option("--fps-limit", type=float, default=8.0, show_default=True, help="Max OCR updates per second")
@click.option("--window", type=str, default="Binary Live", show_default=True)
def live(camera_index: int, bits_per_char: int, encoding: str, invert: bool, mirror: bool, panel_width: int, fps_limit: float, window: str) -> None:
	"""Show live camera with right-side decoded text panel in real time. Press 'q' to quit."""
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		raise click.ClickException(f"Cannot open camera index {camera_index}")

	prev_time = 0.0
	ocr_interval = 1.0 / max(0.1, fps_limit)
	last_text = ""

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			# Optional mirror
			if mirror:
				frame = cv2.flip(frame, 1)
			# Resize for performance if very large
			h, w = frame.shape[:2]
			max_w = 960
			if w > max_w:
				new_h = int(h * (max_w / w))
				frame = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)

			now = time.time()
			if now - prev_time >= ocr_interval:
				bits = ocr_array_to_bits(frame, invert=invert)
				text = bits_to_text(bits, bits_per_char=bits_per_char, encoding=encoding)
				last_text = text
				prev_time = now

			stacked = render_side_panel(frame, last_text, panel_width=panel_width, header="Decoded")
			cv2.imshow(window, stacked)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	cli()
