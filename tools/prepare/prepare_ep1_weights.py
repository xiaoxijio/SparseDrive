"""Build a strict S1 initialization from G5 OCC and SparseDrive stage1 weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def normalized_state(checkpoint):
    state = checkpoint.get('state_dict', checkpoint)
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}


def depth_channel_indices(source_grid, target_grid, context_channels):
    source = torch.arange(*source_grid, dtype=torch.float64)
    target = torch.arange(*target_grid, dtype=torch.float64)
    matches = torch.isclose(target[:, None], source[None, :], rtol=0, atol=1e-7)
    if not bool((matches.sum(1) == 1).all()):
        raise ValueError('Every target depth bin must correspond to one source depth bin')
    indices = matches.long().argmax(1)
    # Context starts AFTER all source depth logits, never after target depth.
    return torch.cat([indices, torch.arange(len(source), len(source) + context_channels)])


def merge_states(target, g5, stage1, source_depth, target_depth, context_channels):
    """Explicit semantic mappings; fail rather than silently skip unexpected mismatches."""
    g5, stage1 = normalized_state(g5), normalized_state(stage1)
    result, loaded, fresh = {}, [], []
    used = {'g5': set(), 'sparsedrive_stage1': set()}
    occ_prefixes = [
        ('occ_img_neck.', 'img_neck.'),
        ('occ_branch.bev_encoder.backbone.', 'img_bev_encoder_backbone.'),
        ('occ_branch.bev_encoder.neck.', 'img_bev_encoder_neck.'),
        ('occ_branch.occ_head.final_conv.', 'occ_head.final_conv.'),
    ]
    untouched = {
        'head.det_head.instance_bank.anchor': 'keep the current 450-anchor initialization',
        'head.det_head.instance_bank.instance_feature': 'keep current 450-slot initial features',
    }
    indices = depth_channel_indices(source_depth, target_depth, context_channels)
    for key, initial in target.items():
        source_name, source_key, conversion = None, key, 'exact'
        if key in untouched:
            reason = untouched[key]
        elif key.startswith('img_neck.lateral_convs.'):
            reason = 'new backbone input channels; keep complete lateral layer fresh, including bias'
        elif key.startswith('occ_branch.occ_head.predicter.'):
            reason = 'hidden width changed 256->128; keep both Linear layers fresh, including bias'
        else:
            reason = None
            if key.startswith('img_backbone.'):
                source_name = 'g5'
            elif key.startswith(('head.det_head.', 'depth_branch.', 'img_neck.fpn_convs.')):
                source_name = 'sparsedrive_stage1'
            elif key.startswith('occ_branch.view_transform.depth_net.'):
                source_name = 'g5'
                source_key = 'img_view_transformer.depth_net.' + key.rsplit('.', 1)[1]
                conversion = 'depth_bins_and_full_context'
            else:
                for dst, src in occ_prefixes:
                    if key.startswith(dst):
                        source_name, source_key = 'g5', src + key[len(dst):]
                        break
            if source_name is None:
                raise ValueError('No initialization policy for target tensor ' + key)
        if reason is not None:
            result[key] = initial.clone()
            fresh.append(dict(target=key, shape=list(initial.shape), reason=reason))
            continue
        source = g5 if source_name == 'g5' else stage1
        if source_key not in source:
            raise ValueError('Missing {} tensor {}'.format(source_name, source_key))
        value = source[source_key]
        original_shape = list(value.shape)
        if conversion != 'exact':
            expected_channels = len(torch.arange(*source_depth)) + context_channels
            if value.shape[0] != expected_channels:
                raise ValueError('Source depth/context channel count does not match its grid: ' + source_key)
            value = value.index_select(0, indices)
        if value.shape != initial.shape:
            raise ValueError('Shape mismatch for {} <- {}:{}: {} vs {}'.format(
                key, source_name, source_key, tuple(initial.shape), tuple(value.shape)))
        result[key] = value.detach().clone().to(dtype=initial.dtype)
        used[source_name].add(source_key)
        loaded.append(dict(target=key, source=source_name, source_key=source_key,
                           source_shape=original_shape, target_shape=list(initial.shape),
                           conversion=conversion))
    report = dict(loaded=loaded, fresh=fresh,
                  unused_source_counts={name: len(state.keys() - used[name])
                                        for name, state in [('g5', g5), ('sparsedrive_stage1', stage1)]},
                  depth_conversion=dict(source_grid=list(source_depth), target_grid=list(target_depth),
                                        source_rows=indices.tolist(), context_channels=context_channels))
    return result, report


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    from mmcv import Config
    from mmdet.models import build_detector
    import projects.mmdet3d_plugin  # register model components
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/ep1/s1_det_occ_b0_nose.py')
    parser.add_argument('--g5', default='ckpt/g5.pth')
    parser.add_argument('--stage1', default='ckpt/sparsedrive_stage1.pth')
    parser.add_argument('--source-depth-grid', nargs=3, type=float, default=[1., 60., 1.])
    parser.add_argument('--output', default='ckpt/s1_det_occ_init.pth')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cfg = Config.fromfile(args.config)
    cfg.model.img_backbone.pretrained = None
    model = build_detector(cfg.model)
    model.init_weights()
    g5 = torch.load(args.g5, map_location='cpu')
    stage1 = torch.load(args.stage1, map_location='cpu')
    source_classes = stage1.get('meta', {}).get('CLASSES')
    if source_classes is not None and list(source_classes) != list(cfg.class_names):
        raise ValueError('Source detection class order differs from S1')
    merged, report = merge_states(model.state_dict(), g5, stage1,
                                  args.source_depth_grid, cfg.model.occ_branch.grid_config.depth,
                                  cfg.model.occ_branch.view_channels)
    model.load_state_dict(merged, strict=True)
    loaded_keys = {entry['target'] for entry in report['loaded']}
    parameter_counts = dict(total=0, transferred=0, fresh=0)
    for name, parameter in model.named_parameters():
        parameter_counts['total'] += parameter.numel()
        parameter_counts['transferred' if name in loaded_keys else 'fresh'] += parameter.numel()
    counts = {name: sum(e['source'] == name for e in report['loaded'])
              for name in ('g5', 'sparsedrive_stage1')}
    summary = dict(loaded_tensors=len(report['loaded']), fresh_tensors=len(report['fresh']),
                   loaded_by_source=counts, parameter_counts=parameter_counts,
                   transferred_parameter_ratio=parameter_counts['transferred'] / parameter_counts['total'])
    report.update(summary=summary, seed=args.seed,
                  sources={name: dict(filename=Path(path).name, sha256=sha256(path))
                           for name, path in [('g5', args.g5), ('sparsedrive_stage1', args.stage1)]},
                  bn_policy='Keep imported backbone/BEV encoder running statistics; affine remains trainable')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=merged, meta=dict(ep1_initialization=report)), output)
    report_path = output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), report=str(report_path), **summary), indent=2))


if __name__ == '__main__':
    main()
