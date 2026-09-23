"""Evaluate saved Stage 1 predictions, one task per process, without loading a model."""
import argparse
import gc
import os

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
import projects.mmdet3d_plugin  # Register project datasets.
from tools.stage1_results import load_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--result-path', required=True)
    parser.add_argument('--task', required=True, choices=['detection', 'tracking', 'map'])
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.model.head.task_config.with_motion_plan:
        parser.error('Expected a Stage 1 configuration')
    output = os.path.join(cfg.work_dir, 'eval_stage1', args.task)
    os.makedirs(output, exist_ok=True)
    dataset = build_dataset(cfg.data.test)
    dataset.work_dir = output
    results = load_results(args.result_path)
    if len(results) != len(dataset):
        raise ValueError('Prediction count does not match the validation dataset')
    print('Loaded {} predictions for {}'.format(len(results), args.task), flush=True)
    if args.task == 'map':
        from projects.mmdet3d_plugin.datasets.evaluation.map.vector_eval import VectorEvaluate
        path = dataset.format_map_results(results, prefix=output)
        config = dataset.eval_config
        del results, dataset
        gc.collect()
        metrics = VectorEvaluate(config).evaluate(path)
    else:
        tracking = args.task == 'tracking'
        dataset.tracking = tracking
        dataset.tracking_threshold = cfg.evaluation.eval_mode.tracking_threshold
        paths, temporary = dataset.format_results(results, jsonfile_prefix=output, tracking=tracking)
        path = paths['img_bbox'] if isinstance(paths, dict) else paths
        del results
        dataset.data_infos = []
        gc.collect()
        if tracking:
            # The dataset helper also creates an unused NuScenes object here.
            # Construct TrackingEval directly to avoid loading its tables twice.
            from nuscenes.eval.tracking.evaluate import TrackingEval
            evaluator = TrackingEval(
                config=dataset.track3d_eval_configs, result_path=path,
                eval_set='mini_val' if dataset.version == 'v1.0-mini' else 'val',
                output_dir=output, verbose=True, nusc_version=dataset.version,
                nusc_dataroot=dataset.data_root)
            metrics = evaluator.main()
        else:
            metrics = dataset._evaluate_single(path, tracking=False)
        if temporary is not None:
            temporary.cleanup()
    mmcv.dump(metrics, os.path.join(output, 'stage1_metrics.json'))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
