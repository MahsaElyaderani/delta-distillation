"""Render saved predictions as four-panel videos; no model dependencies."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

METHODS = ('delta', 'sort', 'flow', 'sort_flow')


def draw_box(image, box, color, label, scale):
    x1, y1, x2, y2 = np.rint(np.asarray(box) * scale).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(image, label, (max(0, x1), max(14, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def main(args):
    selection = json.loads(args.selection.read_text())
    evaluated = set(json.loads((args.results / 'image_ids.json').read_text()))
    predictions = {}
    for method in METHODS:
        predictions[method] = defaultdict(list)
        for row in json.loads((args.results / f'{method}.json').read_text()):
            if row['score'] >= args.score:
                predictions[method][row['image_id']].append(row)
    output = args.output or args.results / 'visuals'
    output.mkdir(parents=True, exist_ok=True)
    writer, previous, offset, segment = None, None, 0, 0
    try:
        for frame in selection['frames']:
            if frame['image_id'] not in evaluated:
                continue
            image = cv2.imread(str(args.frames / frame['file_name']))
            if image is None:
                raise FileNotFoundError(args.frames / frame['file_name'])
            height, width = image.shape[:2]
            if [width, height] != frame['original_size']:
                raise ValueError('Frame size disagrees with selection.')
            scale = args.width / width
            panel_height = max(2, round(height * scale / 2) * 2)
            base = cv2.resize(image, (args.width, panel_height))
            box_scale = np.array([args.width/width, panel_height/height] * 2)
            new_segment = (previous is None or frame['video_id'] != previous['video_id']
                           or frame['frame_id'] != previous['frame_id']+1
                           or frame['original_size'] != previous['original_size'])
            if new_segment:
                if writer is not None:
                    writer.release()
                offset = 0
                segment += 1
                path = output / f"video_{frame['video_id']}_{segment}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.fps, (2*args.width, 2*(panel_height+30)))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create video: {path}')
                print(path, flush=True)
            panels = []
            for method in METHODS:
                panel = base.copy()
                for obj in frame['ground_truth']:
                    box = np.asarray(obj['box_xyxy_normalized']) * [width, height, width, height]
                    draw_box(panel, box, (0, 220, 0), '', box_scale)
                for row in predictions[method][frame['image_id']]:
                    x, y, w, h = row['bbox']
                    name = selection['class_names'][str(row['category_id'])]
                    draw_box(panel, [x, y, x+w, y+h], (0, 140, 255),
                             f"{name} {row['score']:.2f}", box_scale)
                panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
                title = f"{method} | frame {frame['frame_id']}" + (' | KEYFRAME' if offset % 10 == 0 else '')
                cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                panels.append(panel)
            writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
            previous = frame
            offset += 1
    finally:
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--score', type=float, default=.3, help='Display threshold only')
    parser.add_argument('--fps', type=float, default=10, help='Playback speed, not measured inference FPS')
    parser.add_argument('--width', type=int, default=640, help='Width of each panel')
    args = parser.parse_args()
    if not 0 <= args.score <= 1 or args.fps <= 0 or args.width < 2 or args.width % 2:
        parser.error('Use a score in [0,1], positive FPS and a positive even width.')
    main(args)
