import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Cardiomyocyte contractility measurement")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # Parser fo function A
    parser_a = subparsers.add_parser('base', help='Optical flow based method')
    parser_a.add_argument("-i", "--video", nargs='+', help="path to the input video (.mp4/.avi)")
    parser_a.add_argument("-ov", "--outputvideo", required=False, help="path to output video")
    # Parser for function B
    parser_b = subparsers.add_parser('beatprofiler', help='beatProfiler')
    parser_b.add_argument("-i", "--image", nargs='+', help="path to the input video (.mp4/.avi)")
    parser_b.add_argument("-f", "--framerate", help = "Number of frames per second")
    parser_b.add_argument("-m", "--maskmethod", help = "Method for calculating masks (dynamic range, mean fft, max fft)")
    parser_b.add_argument("-r", "--reference", help = "Reference frame selection method for comparison (auto, manual)")
    parser_b.add_argument("-rf", "--refframe", help = "Manual frame selection")

    return parser.parse_args()