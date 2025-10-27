import cv2
import os
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', type=str, nargs='+', required=True,
                    help="Directories of input frame sequence")
parser.add_argument('-o', '--out', type=str, required=True,
                    help="Path to output video")
parser.add_argument('-f', '--fps', type=int, default=24,
                    help="The FPS of output video")
parser.add_argument('-n', '--name', type=str, default='video',
                    help="Name of output video")
args = parser.parse_args()
roots = [root for root in args.root]
nfile = len(roots)
output = args.out
img_roots = [os.path.join(root, 'keys') for root in roots]
fps = args.fps
name = args.name

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_root = os.path.join(output, name + '.mp4')

file_lists = []
for img_root in img_roots:
    file_list = os.listdir(img_root)
    file_list.sort()
    file_lists.append(file_list)
video_len = min([len(file_list) for file_list in file_lists])

ref_frame = cv2.imread(os.path.join(img_roots[0], file_lists[0][0]))
(H, W, C) = ref_frame.shape
if nfile & 1:
    size = (nfile * W, H)
else:
    size = (2 * W, nfile // 2 * H)

#videoWriter = cv2.VideoWriter(video_root, fourcc, fps, size, True)
writer = imageio.get_writer(video_root, format='FFmpeg', fps=fps)

for i in range(video_len):
    frames = []
    for j in range(nfile):
        frames.append(cv2.cvtColor(cv2.imread(os.path.join(img_roots[j], file_lists[j][i])), cv2.COLOR_BGR2RGB))
    if nfile & 1:
        frame = cv2.hconcat(frames)
    else:
        vframes = [cv2.hconcat([frames[k], frames[k+1]]) for k in range(0, nfile, 2)]
        frame = cv2.vconcat(vframes)
    #videoWriter.write(frame)
    writer.append_data(frame)
print('done!\n')
    
#videoWriter.release()
writer.close()  