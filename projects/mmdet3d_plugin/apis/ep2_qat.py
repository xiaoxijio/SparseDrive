"""Opt-in Oritek INT8 convolution islands; geometry, indexing and losses stay FP.

Keep the SDK import lazy so existing floating-point jobs need no QAT package.
"""
import copy
import types
import logging
import torch
from torch import nn
from mmcv.runner import Hook

FORMAT_VERSION = 1


def qat_tool():
    from oritek_qat import quantize_tool
    return quantize_tool


def convolution_paths(model, recipe):
    scopes = tuple(recipe['scopes'])
    paths = [name for name, child in model.named_modules()
             if isinstance(child, nn.Conv2d) and any(
                 name == scope or name.startswith(scope + '.') for scope in scopes)]
    missing = [scope for scope in scopes if not any(
        name == scope or name.startswith(scope + '.') for name in paths)]
    if missing:
        raise ValueError('QAT scopes contain no Conv2d: ' + ', '.join(missing))
    return paths


def _persist_scales(module):
    # SDK 2.1.6 stores scale/zero_point as plain tensors, absent from state_dict
    # and nn.Module.to(). Promote them to buffers AFTER threshold initialization.
    for child in module.modules():
        if not type(child).__module__.startswith('oritek_qat.'):
            continue
        for name in ('scale', 'zero_point'):
            value = getattr(child, name, None)
            if isinstance(value, torch.Tensor) and name not in child._buffers and name not in child._parameters:
                delattr(child, name)
                child.register_buffer(name, value.detach().clone())


def attention_fp16_only(model):
    # Keep SDK fake quant and all other model operations in FP32. Native
    # FlashAttention alone requires half precision. Parameter names unchanged.
    for module in model.modules():
        if type(module).__name__ != 'FlashMHA' or getattr(module, '_qat_half_attention', False):
            continue
        original = module.forward
        def forward(self, q, k, v, key_padding_mask=None, _original=original):
            with torch.cuda.amp.autocast(enabled=True):
                result, weights = _original(q, k, v, key_padding_mask=key_padding_mask)
            return result.float(), weights
        module.forward = types.MethodType(forward, module)
        module._qat_half_attention = True


def convert_convolutions(model, recipe, forward_example):
    """Capture real layer inputs, then convert Conv2d leaves deterministically.

    Leaf islands deliberately leave pooling/index arithmetic, softmax, grid
    sampling, and the OCC Softplus outside the SDK FX tracer/quantizer.
    """
    if hasattr(model, '_qat_format_version'):
        raise ValueError('Model is already converted to QAT')
    if recipe.get('algorithm', 'fake_quantization') != 'fake_quantization':
        raise ValueError('This recipe currently implements fake_quantization INT8 only')
    q = qat_tool()
    q.initialize()
    paths = convolution_paths(model, recipe)
    examples, handles = {}, []
    limit = int(recipe.get('calibration_max_batch', 6))
    if limit < 1:
        raise ValueError('calibration_max_batch must be positive')
    def capture(name):
        def hook(module, args):
            if name not in examples:
                examples[name] = args[0][:limit].detach().float().cpu().clone()
        return hook
    was_training = model.training
    model.eval()
    try:
        for name in paths:
            handles.append(model.get_submodule(name).register_forward_pre_hook(capture(name)))
        with torch.no_grad():
            forward_example()
    finally:
        for handle in handles:
            handle.remove()
    missing = set(paths) - examples.keys()
    if missing:
        raise ValueError('Calibration did not execute QAT layers: ' + ', '.join(sorted(missing)))
    model.cpu()
    log = logging.getLogger('mmdet')
    for i, name in enumerate(paths):
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        original = parent._modules[leaf]
        # Sequential provides a call_module node even when the original is a
        # bare Conv2d; tracing a root Conv2d itself can bypass SDK conversion.
        quant = q.create_quant_model(nn.Sequential(original), fuse_bn=False,
                                     chip_type=recipe.get('chip_type', 'LQ560PlusV200'))
        quant = q.optimize_and_initialize_model(quant, examples.pop(name))
        _persist_scales(quant)
        if not any(type(m).__module__.startswith('oritek_qat.') for m in quant.modules()):
            raise RuntimeError('No SDK quantizers created for ' + name)
        parent._modules[leaf] = quant
        if (i + 1) % 20 == 0:
            log.info('QAT converted %d/%d convolutions', i + 1, len(paths))
    # MMCV's checkpoint collector includes even non-persistent buffers.
    # Make the fixed OCC resampling grid explicit in QAT state so both torch
    # and MMCV checkpoints have the same strict-loading schema.
    model.occ_branch.view_transform._non_persistent_buffers_set.discard('frustum')
    model.register_buffer('_qat_format_version', torch.tensor(FORMAT_VERSION))
    model._oritek_qat_recipe = copy.deepcopy(dict(recipe))
    model.train(was_training)
    # Every eval path, including MMDet EvalHook's direct model.eval(), freezes
    # collectors. No dependence on runner.before_val_iter being dispatched.
    model._qat_observer_mode = None
    def control(module, args):
        observe = model.training and not getattr(model, '_qat_observers_frozen', False)
        # Switch all islands once when train/eval/freeze state changes, rather
        # than recursively writing SDK flags at every convolution invocation.
        if model._qat_observer_mode != observe:
            for name in paths:
                (q.enable_observer if observe else q.disable_observer)(model.get_submodule(name))
            model._qat_observer_mode = observe
    model.register_forward_pre_hook(control)
    return model


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def prepare_training(model, loader, cfg, logger):
    from mmcv.parallel import scatter
    if cfg.get('fp16') is not None:
        raise ValueError('SDK QAT uses FP32 outside native FlashAttention; set fp16=None')
    attention_fp16_only(model)
    if cfg.get('auto_resume'):
        raise ValueError('Use explicit --resume-from for QAT; auto_resume is disabled')
    if any(h.get('type') == 'MEGVIIEMAHook' for h in cfg.get('custom_hooks', [])):
        raise ValueError('Disable MEGVIIEMAHook for QAT quantizer state')
    if not cfg.resume_from:
        if not cfg.load_from:
            raise ValueError('QAT needs a trained FP load_from checkpoint')
        # Fail on missing or mismatched tensors, including migrated OCC weights.
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        if 'oritek_qat' in checkpoint.get('meta', {}):
            raise ValueError('Use resume_from for QAT checkpoints, not FP load_from')
        state = checkpoint.get('state_dict', checkpoint)
        state = {k[7:] if k.startswith('module.') else k: v for k,v in state.items()}
        if 'occ_branch.view_transform.frustum' not in model.state_dict():
            state.pop('occ_branch.view_transform.frustum', None)
        model.load_state_dict(state, strict=True)
        logger.info('Loaded FP weights before QAT conversion: %s', cfg.load_from)
    else:
        checkpoint = torch.load(cfg.resume_from, map_location='cpu')
        validate_recipe(checkpoint, cfg.qat)
    # Do not consume/abandon the training sampler: InfiniteGroupBatchSampler
    # clears its group buffer only AFTER a yielded batch is resumed.
    example = loader.collate_fn([loader.dataset[0]])
    batch = scatter(example, [torch.cuda.current_device()])[0]
    del example
    model.cuda()
    def example():
        # Native FlashAttention requires FP16 Q/K/V. Conv quantizers keep FP32
        # thresholds while MMCV AMP/autocast handles training arithmetic.
        with torch.cuda.amp.autocast(enabled=False):
            return model(**batch)
    model = convert_convolutions(model, cfg.qat, example)
    model.head.det_head.instance_bank.reset()
    del batch
    model.cuda()
    if cfg.resume_from:
        # Strict check now; runner.resume later restores optimizer/scheduler/epoch.
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model._qat_observer_mode = None
    cfg.load_from = None  # runner must not reload FP weights into renamed QAT layers
    return model


def validate_recipe(checkpoint, recipe):
    saved = checkpoint.get('meta', {}).get('oritek_qat')
    if not saved or saved.get('format_version') != FORMAT_VERSION:
        raise ValueError('Expected a QAT checkpoint produced by this implementation')
    for key in ('scopes', 'chip_type', 'algorithm'):
        if saved['recipe'].get(key) != recipe.get(key):
            raise ValueError('QAT checkpoint/config disagree on ' + key)


class OritekQATHook(Hook):
    def __init__(self, recipe):
        self.recipe = copy.deepcopy(dict(recipe))

    def before_run(self, runner):
        from importlib.metadata import version
        # SDK load_state_dict may replace plain/buffer scales with CPU tensors.
        runner.model.cuda(torch.cuda.current_device())
        runner.meta = runner.meta or {}
        runner.meta['oritek_qat'] = dict(
            format_version=FORMAT_VERSION, recipe=self.recipe,
            sdk_version=version('oritek-qat-pytorch'))
        self._sync(runner.model.module, self.recipe.get('coalesced_sync', False))
        runner.logger.info('QAT active: %s', self.recipe)

    def before_train_iter(self, runner):
        model = runner.model.module
        model._qat_observers_frozen = runner.iter >= self.recipe.get('freeze_observer_iter', 879)
        if runner.iter >= self.recipe.get('freeze_bn_iter', 0):
            freeze_bn(model)

    @staticmethod
    def _sync(model, coalesced=False):
        # Keep collector buffers consistent at iteration boundaries across ranks;
        # checkpoint rank 0's scales and min/max, never EMA-average these values.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensors = [tensor for name, tensor in model.state_dict().items()
                       if 'quantizer' in name or name == '_qat_format_version']
            if coalesced and tensors:
                # Same rank-0 values and iteration boundaries; bucket small
                # collector buffers to avoid hundreds of tiny NCCL launches.
                torch.distributed._broadcast_coalesced(
                    torch.distributed.group.WORLD, tensors, 1024 * 1024, 0)
            else:
                for tensor in tensors:
                    torch.distributed.broadcast(tensor, src=0)

    def after_train_iter(self, runner):
        self._sync(runner.model.module, self.recipe.get('coalesced_sync', False))
