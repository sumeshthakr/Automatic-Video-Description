import argparse
import cv2



# construct the argument parse
parser = argparse.ArgumentParser(description='Script to extract videos')

parser.add_argument("--video_id", help="input_id of video")

parser.add_argument("--startframe", help="starting frame")
parser.add_argument("--endframe", help="ending frame")
parser.add_argument("--path", help="path to the video file")
args = parser.parse_args()


def process_video(path,video_id,start,end):
    start_frame = start
    end_frame = end
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(video_id+"_"+str(start_frame)+"_"+str(end_frame)+".avi", fourcc, fps, (w,h))
    
    frame_count = 0
    while frame_count < end_frame:
        ret, frame = cap.read()
        frame_count += 1
        if frame_count >= start_frame:
            out.write(frame)
    cap.release()
    out.release()

if __name__ == '__main__':
        process_video(args.path,args.video_id,args.startframe,args.endframe)