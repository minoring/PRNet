import argparse
import ast


def define_flags():
  parser = argparse.ArgumentParser(
      description=
      'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network'
  )

  parser.add_argument(
      '-i',
      '--inputDir',
      default='TestImages/',
      type=str,
      help='path to the input directory, where input images are stored.')
  parser.add_argument(
      '-o',
      '--outputDir',
      default='TestImages/results',
      type=str,
      help=
      'path to the output directory, where results (obj, text files) will be stored.'
  )
  parser.add_argument('--gpu',
                      default='0',
                      type=str,
                      help='set gpu id, -1 for CPU')
  parser.add_argument(
      '--isDlib',
      default=True,
      type=ast.literal_eval,
      help=
      'whether to use dlib for detecting faces, default is True, if False, the input image should be cropped in advance'
  )
  parser.add_argument(
      '--is3d',
      default=True,
      type=ast.literal_eval,
      help='whether to output 3D face (.obj). default save colors.')
  parser.add_argument(
      '--isMat',
      default=False,
      type=ast.literal_eval,
      help='whether to save vertices, color, triangles as mat for matlab showing'
  )
  parser.add_argument('isKpt',
                      default=False,
                      type=ast.literal_eval,
                      help='whether to output key points (.txt)')
  parser.add_argument('--isPose',
                      default=False,
                      type=ast.literal_eval,
                      help='whether to output estimated pose (.txt)')
  parser.add_argument(
      '--isShow',
      default=False,
      type=ast.literal_eval,
      help='whether to show the results with opencv (need opencv)')
  parser.add_argument('--isImage',
                      default=False,
                      type=ast.literal_eval,
                      help='wheter to save input image')
  parser.add_argument('--isFront',
                      default=False,
                      type=ast.literal_eval,
                      help='whether to frontalize vertices (mesh)')
  parser.add_argument('--isDepth',
                      default=False,
                      type=ast.literal_eval,
                      help='whether to output depth image')
  parser.add_argument('--isTexture',
                      default=False,
                      type=ast.literal_eval,
                      help='whether to save texture in obj file')
  parser.add_argument(
      '--isMask',
      default=False,
      type=ast.literal_eval,
      help=
      'whether to set invisible pixels (due to self-occlusion) in texture as 0')
  parser.add_argument(
      '--texture_size',
      default=256,
      type=int,
      help='size of texture map, default is 256. need isTexture True')

  return parser
