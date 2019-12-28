import os

from api import PRN
from utils.flags import define_flags


def main(args):
  if args.isShow or args.isTexture:
    import cv2
    from utils.cv_plot import plot_kpt, plot_vertices
  
  # ---- init PRN
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
  prn = PRN(is_Dlib=args.isDlib)


if __name__ == '__main__':
  parser = define_flags()
  main(parser.parse_args())
