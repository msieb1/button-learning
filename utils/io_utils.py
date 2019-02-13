import os
from os.path import join
import functools
import imageio
import numpy as np
from PIL import Image
import logging

import sys
import matplotlib.pyplot as plt
import pickle
from pdb import set_trace

def view_image(frame):
    # For debugging. Shows the image
    # Input shape (3, 299, 299) float32
    img = Image.fromarray(np.transpose(frame * 255, [1, 2, 0]).astype(np.uint8))
    img.show()

def write_to_csv(values, keys, filepath):
    if  not(os.path.isfile(filepath)):
        with open(filepath, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open(filepath, 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)


def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)
def write_video(file_name, path, frames):
    imageio.mimwrite(os.path.join(path, file_name), frames, fps=60)

def read_video(filepath, frame_size):
    imageio_video = imageio.read(filepath)
    snap_length = len(imageio_video) 
    frames = np.zeros((snap_length, 3, frame_size.shape[0], frame_size.shape[1]))
    resized = map(lambda frame: resize_frame(frame, frame_size), imageio_video)
    for i, frame in enumerate(resized):
        frames[i, :, :, :] = frame
    return frames

def read_extracted_video(filepath, frame_size):
    try:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))
    except:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    all_images = [file for file in files if file.endswith('.jpg')]
    snap_length = len(all_images) 
    frames = []
    for i, filename in enumerate(all_images):
        frame = plt.imread(os.path.join(filepath, filename))
        frames.append(frame)
    return frames

def read_extracted_rcnn_results(filepath, frame_size):
    try:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))
    except:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_files = [file for file in files if file.endswith('.pkl')]
    snap_length = len(all_files) 
    all_results = []
    for i, filename in enumerate(all_files):
        with open(os.path.join(filepath, filename), 'rb') as fb:
            all_results.append(pickle.load(fb))    
    return all_results

def read_caption(filepath):
    try:
        with open(filepath, 'r') as fp:
            caption = fp.readline()
        return caption
    except:
        print("{} does not exist".format(filepath))
        return None

def ls_directories(path):
    return next(os.walk(path))[1]

# def ls(path):
#     # returns list of files in directory without hidden ones.
#     return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4] == '.mov')], key=lambda x: int(x.split('_')[0] + x.split('.')[0].split('view')[1]))
#     # randomize retrieval for every epoch?

def ls(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4:] == '.mov')], key=lambda x: int(x.split('_')[0]))
    # rand

def ls_unparsed_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-5] != 'd' and p.endswith('.txt')], key=lambda x: int(x.split('.')[0]))


def ls_npy(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-4:] == '.npy'], key=lambda x: x.split('.')[0])
    # rand

def ls_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p.endswith('.txt')], key=lambda x: x.split('.')[0])

def ls_view(path, view):
    # Only lists video files
    return sorted([p for p in os.listdir(path) if p[0] != '.' and (p.endswith(str(view) + '.mp4'))], key=lambda x: int(x.split('_')[0]))

def ls_extracted(path):
     # returns list of folders in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if (p[0] != '.' and p != 'debug') ], key=lambda x: int(x.split('_')[0]))


def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def save_np_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    np.save(filepath, file)

def merge_dicts(x, *args):
    z = x.copy()   # start with x's keys and values
    for y in args:
        z.update(y)    # modifies z with y's keys and values & returns None
    return z

def timer(start, end):
  """Returns a formatted time elapsed."""
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

def create_data_folders(save_path, visual):
    rgb_folder = 'rgb'
    depth_folder = 'depth'
    seg_folder = 'masks'
    flow_folder = 'flow'
    info_folder = 'info'
    vid_folder = 'videos'
    sensor_folder = 'sensor'
    rgb_folder = join(save_path, rgb_folder)
    depth_folder = join(save_path, depth_folder)
    seg_folder = join(save_path, seg_folder)
    flow_folder = join(save_path, flow_folder)
    info_folder = join(save_path, info_folder)
    vid_folder = join(save_path, vid_folder)
    sensor_folder = join(save_path, sensor_folder)

    base_folder = save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if visual is True:
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)
        if not os.path.isdir(depth_folder):
            os.makedirs(depth_folder)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)
        if not os.path.isdir(flow_folder):
            os.makedirs(flow_folder)
        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)
        if not os.path.isdir(vid_folder):
            os.makedirs(vid_folder)
        if not os.path.isdir(sensor_folder):
            os.makedirs(sensor_folder)
        return rgb_folder, depth_folder, seg_folder, flow_folder, info_folder, base_folder, vid_folder, sensor_folder
    else:
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)
        if not os.path.isdir(depth_folder):
            os.makedirs(depth_folder)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)
        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)
        if not os.path.isdir(vid_folder):
            os.makedirs(vid_folder)
        if not os.path.isdir(sensor_folder):
            os.makedirs(sensor_folder)
        return rgb_folder, depth_folder, seg_folder, info_folder, base_folder, vid_folder, sensor_folder




