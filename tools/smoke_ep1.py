"""Exercise real OCC labels, gradients, scene resets and the tensor state API."""
import argparse
import json
import os
import time
import numpy as np
import torch
from mmdet.datasets import build_dataset
from torchvision.ops import SqueezeExcitation
from tools.ep1_common import load_model, sample_to_device, make_export_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', default='work_dirs/ep1/smoke_report.json')
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    cfg, model = load_model(args.config)
    assert not any(isinstance(module, SqueezeExcitation) for module in model.img_backbone.modules())
    dataset = build_dataset(cfg.data.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    report = dict(parameters=sum(p.numel() for p in model.parameters()),
                  component_parameters={name: sum(p.numel() for p in module.parameters())
                                        for name, module in model.named_children()}, training=[])
    model.train()
    start = time.monotonic()
    for step in range(args.steps):
        data = sample_to_device(dataset, step)
        optimizer.zero_grad(set_to_none=True)
        losses = model(**data)
        loss, logs = model._parse_losses(losses)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        for name in ('img_backbone', 'occ_img_neck', 'occ_branch', 'head'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Missing/non-finite gradients in ' + name)
            if not any(torch.count_nonzero(g).item() for g in grads):
                raise RuntimeError('All-zero gradients in ' + name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 25)
        optimizer.step()
        report['training'].append(logs)
        print('step', step, logs, flush=True)
    report['training_seconds'] = time.monotonic() - start
    report['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2

    model.eval()
    bank = model.head.det_head.instance_bank
    bank.reset()
    with torch.no_grad():
        # Normal PyTorch API, including independent per-sample output dictionaries.
        native_first = model(**data)
        assert not bank.mask.any(), 'First frame used history'
        assert native_first[0]['img_bbox']['occ_pred'].shape == (128, 128, 16)
        bank.reset()
        inputs = make_export_inputs(model, data)
        first = model.forward_export(*inputs)
        for output in first:
            assert torch.isfinite(output).all()
        assert first[0].shape == (1, 450, 10)
        assert first[3].shape == (1, 128, 128, 16, 18)
        assert first[4].shape == (1, 300, 256)
        # Invalid nonzero cached data must give exactly the same first-frame outputs.
        dirty = list(inputs)
        dirty[4] = torch.randn_like(dirty[4])
        dirty[5] = torch.randn_like(dirty[5])
        dirty[6] = torch.rand_like(dirty[6])
        for i, (actual, expected) in enumerate(zip(model.forward_export(*dirty), first)):
            # CUDA scatter_add has nondeterministic summation order in OCC.
            tolerance = 1e-4 if i == 3 else 0
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        history = list(inputs)
        history[4:7] = first[4:7]
        history[-1] = torch.ones_like(history[-1])
        second = model.forward_export(*history)
        assert bank.mask.all(), 'Valid history was ignored'
        assert not torch.allclose(first[0], second[0]), 'History did not affect detection'
        # Out-of-range dt must behave like a scene start, including velocity scaling.
        history[-2] = torch.full_like(history[-2], 3.0)
        for i, (actual, expected) in enumerate(zip(model.forward_export(*history), first)):
            tolerance = 1e-4 if i == 3 else 0
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        report['output_shapes'] = [list(x.shape) for x in first]
        report['first_frame_invalid_history_parity'] = True
        report['temporal_frame_changes_detection'] = True
        report['stale_history_resets'] = True
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
