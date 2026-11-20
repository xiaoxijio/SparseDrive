"""Strictly extract a reusable SDK B0-noSE backbone from a BEVDet checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from projects.mmdet3d_plugin.models.ep1.backbone import EfficientNetB0NoSE


def extract_backbone(checkpoint, target):
    raw = checkpoint.get('state_dict', checkpoint)
    source = {}
    for key, value in raw.items():
        key = key[7:] if key.startswith('module.') else key
        if key.startswith('img_backbone.'):
            source[key[len('img_backbone.'):]] = value
    missing = [k for k in target if k not in source]
    unexpected = sorted(source.keys() - target.keys())
    mismatched = [k for k in target if k in source and target[k].shape != source[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError('Backbone mismatch: ' + json.dumps(dict(
            missing=missing, unexpected=unexpected, mismatched=mismatched)))
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='ckpt/g5.pth')
    parser.add_argument('--compare', help='Check another checkpoint for structural compatibility')
    parser.add_argument('--output', default='ckpt/s1_b0_g5.pth')
    args = parser.parse_args()
    model = EfficientNetB0NoSE()
    source = torch.load(args.source, map_location='cpu')
    state = extract_backbone(source, model.state_dict())
    model.load_state_dict(state, strict=True)
    report = dict(source=Path(args.source).name, source_epoch=source.get('epoch'),
                  tensors=len(state), parameters=sum(p.numel() for p in model.parameters()),
                  norm_eval=True, activation='ReLU6', se=False,
                  missing=[], unexpected=[], mismatched=[])
    if args.compare:
        other = torch.load(args.compare, map_location='cpu')
        extract_backbone(other, model.state_dict())
        report['compatible_alternative'] = Path(args.compare).name
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, meta=report), output)
    print(json.dumps(dict(output=str(output), **report), indent=2))


if __name__ == '__main__':
    main()
