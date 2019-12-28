import os

from predictor import PosPrediction


class PRN(object):
  """Joint 3D Face Reconstruction and Dense Alighment with Position Map Regression Network
  """

  def __init__(self, is_Dlib=False, prefix='.'):
    """
    Args:
      is_dlib(bool, optional): if True, dlib is used for detecting faces.
      prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    """
    # Resolution of input and output image size.
    self.resolution_input = 256
    self.resolution_output = 256
    # Load detectors
    if is_Dlib:
      import dlib
      detector_path = os.path.join(
          prefix, 'Data/net-data/mmod_human_face_detector.dat')
      self.face_detector = dlib.cnn_face_detection_model_v1(
          detector_path)  # TODO Invest more and try other face detector.
    # Load PRN
    self.pos_predictor = PosPrediction(self.resolution_input,
                                       self.resolution_output)
    prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
    if not os.path.isfile(prn_path + '.data-00000-of-00001'):
      print('Please download PRN trained model first.')
      exit()
    self.pos_predictor.restore(prn_path)
