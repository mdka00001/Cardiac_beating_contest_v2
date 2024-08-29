from input.input import parse_args
from methods.base import *
import json
from datetime import datetime


def process_video(video_path, file_name=None):
    video_capture = cv2.VideoCapture(video_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frames = []
    ret, frame = video_capture.read()
    while ret:
        frames.append(cv2.resize(frame, (1280, 720)))
        ret, frame = video_capture.read()
    video_capture.release()

    norms = CardiacBeatingBase.save_optical_flow_video(frames, frame_rate, file_name)

    cluster_data = CardiacBeatingBase.get_grid_clusters(norms, file_name)

    smoothed_data, time_axis, peaks = CardiacBeatingBase.get_relative_displacement_graph(cluster_1_data=cluster_data,
                                                                                         frame_rate=frame_rate, 
                                                                                         image_file_name=file_name)
    
    average_bpm = CardiacBeatingBase.get_BPM(time_axis=time_axis, peaks=peaks)

    return average_bpm

def main():

    args = parse_args()

    bpm_values = []

    if args.command == "base":
        for video in args.video:

            print(str.split(str.split(video, "/")[-1], ".")[0])
            
            average_bpm = process_video(video, str.split(str.split(video, "/")[-1], ".")[0])
            bpm_values.append(average_bpm)

    bpmJson = {
        fr"BPM" : bpm_values

                }
    
    print(json.dumps(bpmJson))
    
if __name__ == '__main__':
    main()
