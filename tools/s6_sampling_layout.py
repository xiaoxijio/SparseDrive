"""ONNX export rewrites for grouped S6 camera sampling."""
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def move_sampling_mask(model):
    """Move a 0/1 mask to grouped weights; preserve the original reduction order."""
    m=onnx.shape_inference.infer_shapes(model)
    by={n.name:n for n in m.graph.node}
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in m.graph.value_info}
    insert={};changed=[]
    for n in m.graph.node:
        if n.op_type!='Mul' or not n.name.startswith('/det_head/layers.') or not n.name.endswith('/Mul_3') or n.name.count('/')!=3:
            continue
        base=n.name[:-len('Mul_3')];target=by.get(base+'Mul_5')
        if target is None:continue
        weights=target.input[1];shape=shapes[weights]
        assert len(shape)==6 and shape[-1]==1,(target.name,shape)
        q,cameras,levels,points,groups,_=shape
        mask=n.input[1];n.op_type='Identity';del n.input[1:]
        key=base+'small_mask_shape';out=base+'small_mask';masked=base+'masked_weights'
        m.graph.initializer.append(nh.from_array(np.array([q,cameras,1,points,1,1],np.int64),key))
        insert[target.name]=[h.make_node('Reshape',[mask,key],[out],name=out),
            h.make_node('Mul',[weights,out],[masked],name=masked)]
        target.input[1]=masked;changed.append(base)
    nodes=[]
    for n in m.graph.node:nodes.extend(insert.get(n.name,[]));nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.value_info[:]
    onnx.checker.check_model(m)
    return m,changed


def aggregate_in_sampling_layout(model):
    """Reduce cameras and levels before transposing the sampled features."""
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in model.graph.value_info}
    consumers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    replacements = {}
    obsolete = set()
    for transpose in model.graph.node:
        if transpose.op_type != 'Transpose' or not transpose.name.endswith('/Transpose'):
            continue
        prefix = transpose.name.rsplit('/', 1)[0] + '/'
        if not prefix.startswith('/det_head/layers.') or prefix.count('/') != 3:
            continue
        chain = []
        current = transpose
        for kind in ('Identity', 'Reshape', 'Mul', 'ReduceSum', 'ReduceSum', 'Reshape'):
            users = consumers.get(current.output[0], [])
            if len(users) != 1 or users[0].op_type != kind or users[0].input[0] != current.output[0]:
                break
            current = users[0]
            chain.append(current)
        if len(chain) != 6:
            continue
        mask, reshape, multiply, cameras_sum, levels_sum, output = chain
        perm = next(h.get_attribute_value(a) for a in transpose.attribute if a.name == 'perm')
        if list(perm) != [0, 4, 1, 2, 5, 3]:
            continue
        b, cams, levels, channels, queries, points = shapes[transpose.input[0]]
        q, c, l, p, groups, singleton = shapes[multiply.input[1]]
        if (b, q, c, l, p, singleton) != (1, queries, cams, levels, points, 1):
            continue
        group_width = channels // groups
        stem = prefix + 'sampling_layout_'
        nodes = []

        def constant(name, values):
            key = stem + name
            model.graph.initializer.append(nh.from_array(np.asarray(values, dtype=np.int64), key))
            return key

        def op(kind, name, inputs, out=None, **attrs):
            key = stem + name
            out = out or key
            nodes.append(h.make_node(kind, inputs, [out], name=key, **attrs))
            return out

        def view(name, value, shape, out=None):
            return op('Reshape', name, [value, constant(name + '_shape', shape)], out)

        # GridSample already produces camera, level, channel, query, point.
        # Transpose the grouped scalar weights instead of the 32x larger values.
        weights = view('weights_5d', multiply.input[1], [q, c, l, p, groups])
        weights = op('Transpose', 'weights_permute', [weights], perm=[1, 2, 4, 0, 3])
        weights = view('weights_flat', weights, [c*l*groups, 1, q*p])
        values = view('values_flat', transpose.input[0], [c*l*groups, group_width, q*p])
        values = op('Mul', 'weighted', [values, weights])
        values = view('separate_axes', values, [c, l, channels, q*p])
        axis = constant('sum_axis', [0])
        values = op('ReduceSum', 'sum_cameras', [values, axis], keepdims=0)
        values = op('ReduceSum', 'sum_levels', [values, axis], keepdims=0)
        values = op('Transpose', 'output_permute', [values], perm=[1, 0])
        view('output', values, [1, q, p, channels], output.output[0])
        replacements[output.name] = nodes
        obsolete.update([transpose.name, mask.name, reshape.name, multiply.name,
                         cameras_sum.name, levels_sum.name])
    new_nodes = []
    for node in model.graph.node:
        if node.name not in obsolete:
            new_nodes.extend(replacements.get(node.name, [node]))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model, len(replacements)
