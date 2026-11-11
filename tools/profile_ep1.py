"""Measure EP1 compression candidates on the current GPU, without training.

These timings use random weights and are engineering measurements, not
accuracy comparisons or a prediction of LQ560 NPU performance.
"""
import argparse
import copy
import gc
import json
import logging
import os
import numpy as np
import torch
from mmcv import Config
from mmcv.utils import get_logger
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from tools.ep1_common import sample_to_device, make_export_inputs


def variant_model_config(config, dims, layers, bev_channels):
    model = copy.deepcopy(config)
    model.img_neck.out_channels = dims
    model.depth_branch.embed_dims = dims
    model.occ_branch.bev_channels = list(bev_channels)
    head = model.head.det_head
    head.num_decoder = layers
    head.instance_bank.embed_dims = dims
    head.anchor_encoder.embed_dims = [dims // 2, dims // 8, dims // 8, dims // 4]
    head.temp_graph_model.embed_dims = head.graph_model.embed_dims = dims * 2
    head.norm_layer.normalized_shape = dims
    head.ffn.in_channels = dims * 2
    head.ffn.embed_dims = dims
    head.ffn.feedforward_channels = dims * 4
    head.deformable_model.embed_dims = head.refine_layer.embed_dims = dims
    head.operation_order = (
        ['gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
        + ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'] * (layers - 1))[2:]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', default='work_dirs/ep1/profile_report.json')
    args = parser.parse_args()
    get_logger('mmcv', log_level=logging.WARNING)
    logging.disable(logging.INFO)
    torch.set_num_threads(4)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    data = sample_to_device(dataset)
    del dataset
    gc.collect()
    variants = [
        ('s1_default', 256, 6, (128, 256, 512)),
        ('det_dim128', 128, 6, (128, 256, 512)),
        ('det_dim128_layers3', 128, 3, (128, 256, 512)),
        ('det_dim128_layers3_bev128', 128, 3, (128, 128, 128)),
    ]
    report = dict(gpu=torch.cuda.get_device_name(), batch_size=1,
                  precision='fp32', cudnn_tf32_allowed=torch.backends.cudnn.allow_tf32,
                  matmul_tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                  random_weights=True, measurements=[])
    with torch.no_grad():
        for name, dims, layers, bev_channels in variants:
            torch.manual_seed(0)
            model = build_detector(variant_model_config(cfg.model, dims, layers, bev_channels))
            model.init_weights()
            model.cuda().eval()
            inputs = make_export_inputs(model, data)
            # Warm up a true history frame, including all temporal attention ops.
            first = model.forward_export(*inputs)
            inputs = list(inputs)
            inputs[4:7] = first[4:7]
            inputs[-1] = torch.ones_like(inputs[-1])
            for _ in range(3):
                output = model.forward_export(*inputs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = model.forward_export(*inputs)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            measurement = dict(name=name, parameters=sum(p.numel() for p in model.parameters()),
                               median_ms=float(np.median(timings)), mean_ms=float(np.mean(timings)),
                               peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2)
            report['measurements'].append(measurement)
            print(measurement, flush=True)
            del model, inputs, first, output
            gc.collect()
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(report, handle, indent=2)


if __name__ == '__main__':
    main()
