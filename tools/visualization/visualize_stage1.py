import os
import glob
import argparse
import subprocess
import tempfile
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from tools.stage1_results import load_results
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender

class Stage1BEVRender(BEVRender):
    """BEV renderer without motion/planning labels used only by Stage 2."""

    def _render_command(self, data):
        pass

    def _render_legend(self):
        pass


plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = True, # True: draw history tracked boxes
    motion = False,
    map = True,
    planning = False,
)
START = 0
END = 81
INTERVAL = 1


class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        
        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)
        self.results = load_results(args.result_path)
        self.plot_choices = plot_choices
        self.set_output(args.out_dir)

    def set_output(self, out_dir):
        self.out_dir = out_dir
        self.combine_dir = os.path.join(out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        plot_choices = self.plot_choices
        self.bev_render = Stage1BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]['img_bbox']

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)
    
    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, indices, fps=12, downsample=4):
        # Only use frames rendered in this run, and stream to limit memory use.
        out = None
        try:
            for index in tqdm(indices, desc='Encoding video'):
                path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError('Cannot read image: ' + path)
                height, width = img.shape[:2]
                size = (width // downsample, height // downsample)
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                if out is None:
                    out = cv2.VideoWriter(os.path.join(self.out_dir, 'video.mp4'),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                    if not out.isOpened():
                        raise RuntimeError('Cannot open video writer')
                out.write(img)
        finally:
            if out is not None:
                out.release()


def group_scenes(data_infos):
    """Group global dataset indices by scene, preserving dataset order."""
    scenes = {}
    for index, info in enumerate(data_infos):
        scenes.setdefault(info['scene_token'], []).append(index)
    return list(scenes.items())


def concat_videos(video_paths, output_path):
    """Join compatible scene videos without re-encoding them."""
    if not video_paths:
        raise ValueError('No scene videos found to concatenate')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    list_path = None
    temporary_output = output_path + '.tmp.mp4'
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', prefix='scene_videos_',
                                         dir=os.path.dirname(output_path), delete=False) as listing:
            list_path = listing.name
            for path in video_paths:
                absolute = os.path.abspath(path).replace("'", "'\\''")
                listing.write("file '{}'\n".format(absolute))
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-f', 'concat', '-safe', '0', '-i', list_path,
                        '-c', 'copy', '-movflags', '+faststart', temporary_output], check=True)
        os.replace(temporary_output, output_path)
    finally:
        if list_path and os.path.exists(list_path):
            os.remove(list_path)
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
    print('Combined {} scene videos: {}'.format(len(video_paths), output_path))


def existing_scene_videos(out_dir):
    return [os.path.join(out_dir, name, 'video.mp4')
            for name in sorted(os.listdir(out_dir)) if name.startswith('scene_')
            and os.path.isfile(os.path.join(out_dir, name, 'video.mp4'))]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Stage 1 detection, tracking, and map results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='vis',
        help='directory where visualize results will be saved')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--num-scenes', type=int, help='Render first N complete scenes')
    selection.add_argument('--concat-only', action='store_true',
                           help='Join all existing scene videos without rendering')
    selection.add_argument('--scene-indices', type=int, nargs='+',
                           help='Render selected scenes by zero-based index')
    parser.add_argument('--list-scenes', action='store_true', help='List scenes without loading predictions')
    parser.add_argument('--no-track', action='store_true', help='Hide historical boxes')
    parser.add_argument('--interval', type=int, default=INTERVAL, help='Render every Nth frame')
    args = parser.parse_args()
    if args.interval < 1 or (args.num_scenes is not None and args.num_scenes < 1):
        parser.error('--interval and --num-scenes must be positive')
    if not args.list_scenes and not args.concat_only and not args.result_path:
        parser.error('--result-path is required for rendering')

    return args

def main():
    args = parse_args()
    if args.concat_only:
        concat_videos(existing_scene_videos(args.out_dir),
                      os.path.join(args.out_dir, 'all_scenes.mp4'))
        return
    if args.list_scenes:
        cfg = Config.fromfile(args.config)
        dataset = build_dataset(cfg.data.val)
        for number, (token, indices) in enumerate(group_scenes(dataset.data_infos)):
            print(f'{number}: {token} frames={len(indices)} indices={indices[0]}..{indices[-1]}')
        return

    choices = dict(plot_choices)
    if args.no_track:
        choices['track'] = False
    visualizer = Visualizer(args, choices)
    if len(visualizer.results) != len(visualizer.dataset):
        raise ValueError('Prediction count does not match validation dataset')
    scenes = group_scenes(visualizer.dataset.data_infos)
    if args.num_scenes is not None:
        if args.num_scenes > len(scenes):
            raise ValueError(f'Only {len(scenes)} scenes available')
        selected = list(range(args.num_scenes))
    else:
        selected = args.scene_indices

    if selected is None:
        indices = list(range(START, min(END, len(visualizer.results)), args.interval))
        for idx in tqdm(indices):
            visualizer.add_vis(idx)
        visualizer.image2video(indices)
    else:
        selected = list(dict.fromkeys(selected))
        if any(i < 0 or i >= len(scenes) for i in selected):
            raise ValueError(f'Scene indices must be between 0 and {len(scenes) - 1}')
        video_paths = []
        for number in selected:
            token, indices = scenes[number]
            indices = indices[::args.interval]
            visualizer.set_output(os.path.join(args.out_dir, f'scene_{number:03d}_{token}'))
            print(f'Rendering scene {number}: {token}, {len(indices)} frames')
            for idx in tqdm(indices):
                visualizer.add_vis(idx)
            visualizer.image2video(indices)
            video_paths.append(os.path.join(visualizer.out_dir, 'video.mp4'))
        concat_videos(video_paths, os.path.join(args.out_dir, 'all_scenes.mp4'))

if __name__ == '__main__':
    main()