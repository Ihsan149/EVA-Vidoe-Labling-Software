import cv2
import numpy as np

def get_img_size_from_buffer(buffer, flags=cv2.IMREAD_COLOR):
    np_buf = np.frombuffer(buffer.read(), dtype=np.uint8)
    buffer.seek(0)
    img = cv2.imdecode(np_buf, flags)
    shape = img.shape if img is not None else (0, 0, 0)
    return shape
