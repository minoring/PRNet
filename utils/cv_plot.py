import numpy as np
import cv2

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1  # ???


def plot_kpt(image, kpt):
  """Draw 68 key points

  Args:
    image: the input image
    kpt: Numpy array of (68, 3) shape.
  """
  image = image.copy()
  kpt = np.round(kpt).astype(np.int32)
  for i in range(kpt.shape[0]):
    st = kpt[i, :2]
    cv2.circle(image,
               center_cordinates=(st[0], st[1]),
               radius=1,
               color=(0, 0, 255),
               thickness=2)
    if i in end_list:
      continue
    ed = kpt[i + 1, :2]
    cv2.line(image,
             start_point=(st[0], st[1]),
             end_point=(ed[0], ed[1]),
             color=(255, 255, 255),
             thickness=1)


def plot_vertices(image, vertices):
  image = image.copy()
  vertices = np.round(vertices).astype(np.int32)
  for i in range(0, vertices.shape[0], 2):
    st = vertices[i, :2]
    image = cv2.circle(image, (st[0], st[1]), 1, (255, 0, 0), -1)
