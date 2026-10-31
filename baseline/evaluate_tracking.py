"""Compare Delta Distillation and three keyframe-only tracking ablations."""
import argparse
import copy
from contextlib import ExitStack
import json
import inspect
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Resolve the external checkout before importing its model code. The default
# keeps existing commands working when benchmarks lives inside the checkout.
root_parser = argparse.ArgumentParser(add_help=False)
root_parser.add_argument('--deltadistill-path', type=Path, default=Path(__file__).resolve().parents[1],
                         help='Delta Distillation checkout containing lib/ and config/')
ROOT = root_parser.parse_known_args()[0].deltadistill_path.expanduser().resolve()
if not (ROOT / 'lib/models/delta_distillation/conv.py').is_file():
    root_parser.error('Set --deltadistill-path to the Delta Distillation project directory.')
sys.path.insert(0, str(ROOT))
from lib.datasets.imagenet_vid.transforms import transforms_coco_eval
from lib.models.delta_distillation.convertor import convert_convs
from lib.models.efficientdet.effdet import EfficientDet, DetBenchPredict, get_efficientdet_config
from lib.utils.config import Config
from lib.utils.nn import set_delta_distillation_schedule
from tracking import track_clip

METHODS = ('delta', 'sort', 'flow', 'sort_flow')
SELECTION = Path(__file__).resolve().parent / 'selection.json'


def load_model(checkpoint, device):
    """Reproduce get_efficientdet(test), without downloads or Faster-RCNN imports."""
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = 30
    model = DetBenchPredict(EfficientDet(config, pretrained_backbone=False), config)
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    layers = [(n, m) for n, m in layers[1:-2]
              if '.se.' not in n and m.groups != m.in_channels]
    cfg = Config({'delta_distillation': {
        'factor_ch': 16, 'channel_reduction_type': 'min', 'gate_bias_init': -1}})
    convert_convs(model, cfg, layers)
    # Older PyTorch lacks weights_only; use its loader for the trusted checkpoint.
    load_options = {'weights_only': True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    state = torch.load(checkpoint, map_location='cpu', **load_options)
    state = state.get('state_dict', state)
    # str.removeprefix is unavailable on the original Python 3.8 environment.
    for prefix in ('module.', 'model.'):
        state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    # Released checkpoint stores duplicate selected-student weights and omits
    # zero student biases. Validate aliases; retain strict checks on all else.
    for key in list(state):
        if '.g_W.' in key:
            prefix, suffix = key.split('.g_W.')
            branch = 'g_Wa' if state[prefix+'.g_gate.bias'] > 0 else 'g_Wb'
            if not torch.equal(state[key], state[prefix+'.'+branch+'.'+suffix]):
                raise ValueError(f'Inconsistent selected-student alias: {key}')
            del state[key]
    for key, value in model.model.state_dict().items():
        if key not in state and (key.endswith('.g_Wa.bias') or key.endswith('.g_Wb.1.bias')):
            state[key] = torch.zeros_like(value)
    model.model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_selection(path):
    """Only selected VID frames; COCO is an evaluation API, not our dataset."""
    selection = json.loads(path.read_text())
    images, annotations, clips = [], [], []
    previous = None
    for frame in selection['frames']:
        width, height = frame['original_size']
        info = dict(id=frame['image_id'], file_name=frame['file_name'],
                    width=width, height=height)
        images.append(info)
        # Reset on video changes, gaps, and every ten frames. Never use unseen frames.
        if (previous is None or frame['video_id'] != previous['video_id']
                or frame['frame_id'] != previous['frame_id'] + 1 or len(clips[-1]) == 10):
            clips.append([])
        clips[-1].append(info)
        previous = frame
        for obj in frame['ground_truth']:
            x1, y1, x2, y2 = np.asarray(obj['box_xyxy_normalized']) * [width, height, width, height]
            annotations.append(dict(id=len(annotations)+1, image_id=info['id'],
                                    category_id=obj['dataset_label'], bbox=[x1, y1, x2-x1, y2-y1],
                                    area=(x2-x1)*(y2-y1), iscrowd=0))
    if len({image['id'] for image in images}) != len(images):
        raise ValueError('Selection contains duplicate image IDs.')
    ground_truth = COCO()
    ground_truth.dataset = dict(images=images, annotations=annotations, categories=[
        dict(id=int(key), name=value) for key, value in selection['class_names'].items()])
    ground_truth.createIndex()
    return ground_truth, clips


def convolution_flops(model, period, device):
    """Executed Conv2d arithmetic only, 2 FLOPs/MAC; excludes tracker and NMS."""
    total = 0

    def count(layer, inputs, output):
        nonlocal total
        kernel = np.prod(layer.kernel_size) * layer.in_channels // layer.groups
        total += output.numel() * (2 * int(kernel) + int(layer.bias is not None))

    hooks = [layer.register_forward_hook(count) for layer in model.modules()
             if isinstance(layer, torch.nn.Conv2d)]
    try:
        set_delta_distillation_schedule(model, period)
        size = model.config.image_size
        model.model(torch.zeros(period, 3, size, size, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    return total


def load_clip(infos, frames_root, transform):
    images, scales, sizes = [], [], []
    for info in infos:
        with Image.open(frames_root / info['file_name']) as source:
            image = source.convert('RGB')
            size = image.size
            if size != (info['width'], info['height']):
                raise ValueError(f"Image dimensions disagree with annotations: {info['file_name']}")
            pixels, target = transform(image, {'img_size': size})
        images.append(pixels)
        scales.append(target['img_scale'])
        sizes.append(size)
    if len(set(sizes)) != 1:
        raise ValueError('Tracking assumes constant image dimensions within each clip.')
    return np.stack(images), np.array(scales), np.array(sizes)


def evaluate(coco, predictions, image_ids):
    # COCO.loadRes cannot handle an empty list. An empty result COCO can.
    if predictions:
        detections = coco.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {**copy.deepcopy(coco.dataset), 'annotations': []}
        detections.createIndex()
    evaluator = COCOeval(coco, detections, 'bbox')
    evaluator.params.imgIds = image_ids  # Includes frames with zero predictions.
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {'AP': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1])}


def records(detections, info, threshold):
    """All methods use original-image COCO xywh boxes and identical filtering."""
    result = []
    for x, y, w, h, score, category in detections:
        if score < threshold or not np.isfinite([x, y, w, h, score]).all():
            continue
        x1, x2 = np.clip([x, x+w], 0, info['width'])
        y1, y2 = np.clip([y, y+h], 0, info['height'])
        if x2 <= x1 or y2 <= y1:
            continue
        result.append(dict(image_id=info['id'], bbox=list(map(float, [x1, y1, x2-x1, y2-y1])),
                           score=float(score), category_id=int(category)))
    return result


@torch.inference_mode()
def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    import cv2
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(0)
    torch.manual_seed(0)
    coco, clips = load_selection(args.selection)
    if args.max_clips:
        clips = clips[:args.max_clips]
    if not clips:
        raise ValueError('No evaluation clips found.')
    model = load_model(args.checkpoint, device)
    teacher_flops = convolution_flops(model, 1, device)
    delta_flops = convolution_flops(model, 10, device)
    transform = transforms_coco_eval(model.config.image_size, use_prefetcher=True)
    seconds = {name: 0. for name in METHODS}
    image_ids = []
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

    def detect(images, scales, sizes, period):
        set_delta_distillation_schedule(model, period)
        tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32)
        tensor = (tensor - mean * 255) / (std * 255)
        return model(tensor, torch.as_tensor(scales, device=device, dtype=torch.float32),
                     torch.as_tensor(sizes, device=device, dtype=torch.float32)).cpu().numpy()

    def infer(images, scales, sizes):
        # Pad only for delta's T=10 tensor layout; never score duplicate frames.
        pad = 10 - len(images)
        padded = np.concatenate([images, np.repeat(images[-1:], pad, axis=0)])
        padded_scales = np.concatenate([scales, np.repeat(scales[-1:], pad)])
        padded_sizes = np.concatenate([sizes, np.repeat(sizes[-1:], pad, axis=0)])
        start = time.perf_counter()
        delta = detect(padded, padded_scales, padded_sizes, 10)[:len(images)]
        delta_time = time.perf_counter() - start
        start = time.perf_counter()
        detect(images[:1], scales[:1], sizes[:1], 1)  # Measure standalone teacher cost.
        teacher_time = time.perf_counter() - start
        # Delta's first frame uses the full teacher. Share its detections exactly:
        # GPU batch-size roundoff can reorder top-k/NMS results, so independent
        # one-frame and ten-frame outputs cannot be compared row by row.
        key = delta[0].copy()
        key = key[(key[:, 4] >= args.score_threshold) & (key[:, 2] > 0) & (key[:, 3] > 0)]
        key[:, :4] /= scales[0]  # Original-image -> resized detector-input pixels.
        outputs, costs = {'delta': delta}, {'delta': delta_time}
        for name in METHODS[1:]:
            start = time.perf_counter()
            tracked = track_clip(images, key, name)
            for t, frame in enumerate(tracked):
                frame[:, :4] *= scales[t]
            outputs[name] = tracked
            costs[name] = teacher_time + time.perf_counter() - start
        return outputs, costs

    args.output.mkdir(parents=True, exist_ok=True)
    # Stream COCO JSON so full validation does not retain four prediction sets.
    with ExitStack() as stack:
        files = {name: stack.enter_context((args.output / f'{name}.json').open('w')) for name in METHODS}
        separators = {name: '' for name in METHODS}
        for file in files.values():
            file.write('[')
        for index, infos in enumerate(clips):
            images, scales, sizes = load_clip(infos, args.frames, transform)
            if index == 0:
                infer(images, scales, sizes)  # One untimed warmup for every path.
            outputs, costs = infer(images, scales, sizes)
            image_ids.extend(info['id'] for info in infos)
            for name in METHODS:
                seconds[name] += costs[name]
                for info, detections in zip(infos, outputs[name]):
                    for record in records(detections, info, args.score_threshold):
                        files[name].write(separators[name] + json.dumps(record))
                        separators[name] = ',\n'
            if index % 10 == 0:
                print(f'Clips: {index+1}/{len(clips)}', flush=True)
        for file in files.values():
            file.write(']')

    summary = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'frames': len(image_ids), 'clips': len(clips), 'results': {},
               'flops_scope': 'Convolution FLOPs only, 2 FLOPs/MAC plus bias additions. '
                              'Excludes optical flow, Kalman, activations and NMS; not total pipeline FLOPs. '
                              'Includes actual padded delta computation on short clips.',
               'timing': 'Warmup excluded; includes normalization, transfer, inference/postprocess, '
                         'and tracking. Excludes image loading/resizing, JSON and scoring. '
                         'Delta uses clip batching; not a streaming latency measurement.'}
    for name in METHODS:
        with (args.output / f'{name}.json').open() as file:
            predictions = json.load(file)
        metrics = evaluate(coco, predictions, image_ids)
        del predictions
        metrics['ms_per_frame'] = 1000 * seconds[name] / len(image_ids)
        metrics['FPS'] = len(image_ids) / seconds[name]
        metrics['neural_conv_GFLOPs_per_frame'] = (
            (delta_flops if name == 'delta' else teacher_flops) * len(clips) / len(image_ids) / 1e9)
        metrics['mAP50'] = metrics.pop('AP50')
        summary['results'][name] = metrics
    (args.output / 'image_ids.json').write_text(json.dumps(image_ids))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['results'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, parents=[root_parser])
    parser.add_argument('--frames', type=Path, required=True, help='ImageNet VID root containing val/')
    parser.add_argument('--selection', type=Path, default=SELECTION, help='Selected VID frames and ground truth')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/test/efficientdet_d0.pth')
    parser.add_argument('--output', type=Path, default=ROOT/'exp/tracking_baselines')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--max-clips', type=int, default=0, help='0 = all selected frames; positive = smoke test')
    parser.add_argument('--threads', type=int, default=1)
    arguments = parser.parse_args()
    if not 0 <= arguments.score_threshold <= 1 or arguments.max_clips < 0 or arguments.threads < 1:
        parser.error('Invalid score threshold, clip limit or thread count.')
    run(arguments)
