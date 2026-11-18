"""Export a fixed-rig S1 graph, retaining dynamic history pose and time."""
import argparse
import json
import warnings
import sys

# This notice concerns a future MMCV release, not this pinned environment.
warnings.filterwarnings("ignore", message=r"On January 1, 2023, MMCV will release.*", category=UserWarning)
from pathlib import Path
import numpy as np
import torch
from mmdet.datasets import build_dataset
from tools.ep1_common import load_model, sample_to_device, make_export_inputs
from projects.mmdet3d_plugin.models.ep1.deployment import (
    FixedRigDeployment, INPUT_NAMES, OUTPUT_NAMES)

def align_cache_rows(expected, actual):
    """Compare jointly permuted TopK cache rows without changing model outputs.

    Tiny backend rounding differences can exchange nearly tied confidence ranks.
    Match encoded anchors one-to-one, then check features and confidence using
    that SAME assignment. Different selected targets still fail allclose.
    """
    from scipy.optimize import linear_sum_assignment
    aligned = list(actual[:4]) + [x.copy() for x in actual[4:]]
    for batch in range(expected[5].shape[0]):
        distance = np.linalg.norm(expected[5][batch, :, None] - actual[5][batch, None, :], axis=-1)
        rows, columns = linear_sum_assignment(distance)
        for index in (4, 5, 6):
            aligned[index][batch, rows] = actual[index][batch, columns]
    return aligned

def main():
    import onnx
    import onnxruntime as ort
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/ep1/s1_det_occ_b0_nose.py')
    parser.add_argument('--checkpoint', default='ckpt/iter_43900.pth')
    parser.add_argument('--calibration', help='npz with projection_mat, image_wh, occ_img2ego')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output', default='onnx/s1.onnx')
    parser.add_argument('--verbose-warnings', action='store_true', help='Print each export warning as well as recording it')
    parser.add_argument('--check-only', action='store_true', help='Validate an existing ONNX without exporting again')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false',
                        help='Disable default onnxsim simplification for debugging')
    args = parser.parse_args()
    if args.simplify:
        try:
            from onnxsim import simplify
        except ImportError:
            parser.error('Default simplification requires onnxsim; install it with python -m pip install onnxsim, or use --no-simplify for debugging')
    torch.manual_seed(0); torch.set_num_threads(4)
    cfg, model = load_model(args.config, args.checkpoint, device='cpu', export_checkpoint=True)
    model.eval()
    dataset = build_dataset(cfg.data.val)
    data = sample_to_device(dataset, args.sample_index, device=-1)
    original = list(make_export_inputs(model, data))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if args.check_only and not args.calibration:
        args.calibration = str(output.with_suffix('.calibration.npz'))
    if args.calibration:
        with np.load(args.calibration, allow_pickle=False) as source:
            for index, key in [(1, 'projection_mat'), (2, 'image_wh'), (3, 'occ_img2ego')]:
                value = torch.from_numpy(source[key]).float()
                if value.shape != original[index].shape:
                    raise ValueError('Unexpected calibration shape for ' + key)
                original[index] = value
            calibration_source = str(source['provenance'].item()) if 'provenance' in source else str(args.calibration)
    else:
        calibration_source = 'nuScenes sample {} ({}) -- PREFLIGHT ONLY'.format(
            args.sample_index, dataset.data_infos[args.sample_index]['token'])
    np.savez(output.with_suffix('.calibration.npz'),
             projection_mat=original[1].numpy(), image_wh=original[2].numpy(),
             occ_img2ego=original[3].numpy(), provenance=np.array(calibration_source))
    wrapper = FixedRigDeployment(model, *original[1:4]).eval()
    inputs = [original[0]] + original[4:]
    inputs[-1] = inputs[-1].float()
    report = dict(calibration=calibration_source, checkpoint=args.checkpoint,
                  initialization=cfg.get('load_from'), inputs=INPUT_NAMES, outputs=OUTPUT_NAMES,
                  host_validity_required=True, time_interval_dynamic=True,
                  npu_compilation_tested=False)
    if not args.check_only:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with torch.no_grad():
                torch.onnx.export(wrapper, tuple(inputs), str(output), opset_version=16,
                                  input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                                  do_constant_folding=True)
        warning_rows = [dict(category=w.category.__name__, message=str(w.message),
                             filename=w.filename, lineno=w.lineno) for w in captured]
        warning_path = output.with_suffix('.warnings.json')
        warning_path.write_text(json.dumps(warning_rows, indent=2)+'\n')
        report['export_warnings'] = dict(count=len(warning_rows), path=str(warning_path))
        if warning_rows:
            print('Export produced {} warnings; full details: {}'.format(len(warning_rows), warning_path), flush=True)
            for category in sorted(set(w['category'] for w in warning_rows)):
                unique = list(dict.fromkeys(w['message'] for w in warning_rows if w['category'] == category))
                print('  {}: {} distinct warning messages'.format(category,len(unique)), flush=True)
            if args.verbose_warnings:
                for w in captured:
                    sys.stderr.write(warnings.formatwarning(w.message,w.category,w.filename,w.lineno))
    graph = onnx.load(str(output))
    for node in graph.graph.node:
        if node.op_type == 'Resize' and any(a.name == 'mode' and a.s == b'linear' for a in node.attribute):
            for a in node.attribute:
                if a.name == 'coordinate_transformation_mode' and a.s == b'half_pixel':
                    a.s = b'pytorch_half_pixel'
    if args.simplify:
        graph, checked = simplify(graph)
        assert checked
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    assert [x.name for x in session.get_inputs()] == INPUT_NAMES
    assert not any(n.op_type in ('And','Not','LessOrEqual','GreaterOrEqual','ScatterElements','If') for n in graph.graph.node)

    def check(name, values):
        full = [values[0]] + original[1:4] + list(values[1:-1]) + [values[-1] > .5]
        with torch.no_grad():
            ref = list(model.forward_export(*full))
            ref[3] = ref[3].argmax(-1).to(torch.int32)
            ref = [x.numpy() for x in ref]
            fixed = [x.numpy() for x in wrapper(*values)]
        actual = session.run(None, dict(zip(INPUT_NAMES, [x.numpy() for x in values])))
        result = {}
        for label, expected, observed in [('source_vs_adapter',ref,fixed), ('adapter_vs_ort',fixed,actual)]:
            aligned = align_cache_rows(expected, observed)
            errors = {}
            for key, a, b in zip(OUTPUT_NAMES, expected, aligned):
                if key == 'occ_classes':
                    errors[key] = int(np.count_nonzero(a != b))
                    if not np.array_equal(a,b):
                        report.setdefault('validation_failures',[]).append(name+'/'+label+'/'+key)
                else:
                    errors[key] = float(np.abs(a-b).max())
                    if not np.allclose(a,b,rtol=3e-3,atol=3e-3):
                        report.setdefault('validation_failures',[]).append(name+'/'+label+'/'+key)
            result[label] = errors
        report[name] = result
        print(name, result, flush=True)
        return actual

    first = check('first', inputs)
    history = list(inputs)
    history[1:4] = [torch.from_numpy(x) for x in first[4:]]
    history[-1] = torch.ones(1)
    history[-2] = torch.tensor([.7])
    history[-3] = history[-3].clone(); history[-3][:,0,3] = -.3
    check('history_nondefault_dt', history)
    history[-1] = torch.zeros(1)
    reset = check('reset', history)
    for a,b in zip(first,reset): np.testing.assert_array_equal(a,b)
    report['output_shapes'] = [list(x.shape) for x in first]
    report['occ_output_bytes'] = int(first[3].nbytes)
    report['validation_passed'] = not report.get('validation_failures')
    report['ort_graph_optimization'] = 'disabled for direct numerical reference'
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved',output,flush=True)
    if not report['validation_passed']:
        raise AssertionError('Numerical validation failed; inspect the JSON report. Tolerances were not relaxed.')


if __name__ == '__main__':
    main()
