"""Export one static ONNX graph and check first/history frames against PyTorch.

Use --checkpoint for a trained EP1 checkpoint. --random-init is only for graph
validation; it does not produce a trained detector or a deployment accuracy.
"""
import argparse
import json
import os
import numpy as np
import torch
from mmdet.datasets import build_dataset
from tools.ep1_common import (load_model, sample_to_device, make_export_inputs,
                               ExportWrapper, INPUT_NAMES, OUTPUT_NAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--random-init', action='store_true')
    parser.add_argument('--output', default='work_dirs/ep1/s1_det_occ_b0_nose.onnx')
    args = parser.parse_args()
    import onnx
    import onnxruntime as ort
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg, model = load_model(args.config, args.checkpoint)
    model.eval()
    dataset = build_dataset(cfg.data.val)
    data = sample_to_device(dataset)
    inputs = make_export_inputs(model, data)
    wrapper = ExportWrapper(model).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, args.output, opset_version=16,
                          input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                          do_constant_folding=True)
    graph = onnx.load(args.output)
    resize_count = 0
    for node in graph.graph.node:
        if node.op_type == 'Resize':
            for attr in node.attribute:
                if attr.name == 'coordinate_transformation_mode':
                    if attr.s == b'half_pixel':
                        # All EP1 Resize output spatial dimensions exceed 1.
                        # These modes are numerically identical in this domain;
                        # emit the exact attribute requested by the NPU guide.
                        attr.s = b'pytorch_half_pixel'
                    if attr.s != b'pytorch_half_pixel':
                        raise RuntimeError('Unexpected EP1 Resize mode: ' + str(attr.s))
                    resize_count += 1
    assert resize_count > 0
    assert not any(node.op_type == 'If' for node in graph.graph.node)
    onnx.checker.check_model(graph)
    onnx.save(graph, args.output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(args.output, sess_options=options,
                                   providers=['CPUExecutionProvider'])
    assert [x.name for x in session.get_inputs()] == INPUT_NAMES

    def check(label, tensors):
        with torch.no_grad():
            expected = [x.cpu().numpy() for x in wrapper(*tensors)]
        feed = {name: tensor.cpu().numpy() for name, tensor in zip(INPUT_NAMES, tensors)}
        actual = session.run(None, feed)
        errors = {}
        for name, reference, result in zip(OUTPUT_NAMES, expected, actual):
            np.testing.assert_allclose(result, reference, rtol=3e-3, atol=3e-3,
                                       err_msg=label + '/' + name)
            errors[name] = float(np.abs(result - reference).max())
        print(label, errors, flush=True)
        return actual, errors

    first, first_errors = check('first_frame', inputs)
    following = list(inputs)
    following[4:7] = [torch.from_numpy(x).to(inputs[0].device) for x in first[4:7]]
    following[-1] = torch.ones_like(following[-1])
    following[7] = following[7].clone()
    following[7][:, 0, 3] = -0.3  # exercise non-identity ego-motion compensation
    second, second_errors = check('history_frame', following)
    assert not np.allclose(first[0], second[0]), 'ONNX history inputs are ineffective'
    # Check the SAME serialized graph resets without tracing another model.
    following[-1] = torch.zeros_like(following[-1])
    _, reset_errors = check('scene_reset', following)
    reset_feed = {name: x.cpu().numpy() for name, x in zip(INPUT_NAMES, following)}
    reset = session.run(None, reset_feed)
    for reference, result in zip(first, reset):
        np.testing.assert_allclose(result, reference, rtol=0, atol=0)
    report = dict(checkpoint=args.checkpoint, random_init=args.random_init,
                  opset=16, resize_pytorch_half_pixel_count=resize_count,
                  inputs=INPUT_NAMES, outputs=OUTPUT_NAMES,
                  output_shapes=[list(x.shape) for x in first],
                  first_frame_max_abs_errors=first_errors,
                  history_frame_max_abs_errors=second_errors,
                  scene_reset_max_abs_errors=reset_errors,
                  npu_compilation_tested=False)
    with open(args.output + '.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    print('Saved one graph:', args.output, flush=True)


if __name__ == '__main__':
    main()
