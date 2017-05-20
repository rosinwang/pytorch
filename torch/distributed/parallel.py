import sys
import threading

import torch
from torch import nn
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from .collectives import all_reduce, get_num_processes
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


_default_streams = []
_reduction_streams = []
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        _default_streams.append(torch.cuda.current_stream())
        _reduction_streams.append(torch.cuda.Stream())

def _thread_fn():
    # TODO: this assumes GPU0 is used to store DDP's original module
    # TODO: this assumes someone is using all visible devices
    def _process_batch():
        dev_grad_batch, dev_events, job_event, is_last = _reduction_queue.get()
        dev_coalesced = []
        dev_coalesced_out = []
        # Coalesce the tensors on all devices
        for i, (grad_batch, cuda_event, reduce_stream) in enumerate(zip(dev_grad_batch, dev_events, _reduction_streams)):
            with torch.cuda.device(i), torch.cuda.stream(reduce_stream):
                reduce_stream.wait_event(cuda_event)
                coalesced = _flatten_tensors(grad_batch)
                dev_coalesced.append(coalesced)
                dev_coalesced_out.append(coalesced.new().resize_as_(coalesced))
                if i == 0:
                    dev_coalesced_out[0].zero_()
        nccl.reduce(dev_coalesced, outputs=dev_coalesced_out, root=0, streams=_reduction_streams)
        # From now on we're only going to work on GPU0
        grad_batch = dev_grad_batch[0]
        coalesced = dev_coalesced_out[0]
        reduce_stream = _reduction_streams[0]
        with torch.cuda.stream(reduce_stream):
            coalesced *= (1. / get_num_processes())
            all_reduce(coalesced)
            for grad, reduced in zip(grad_batch, _unflatten_tensors(coalesced, grad_batch)):
                grad.copy_(reduced)
        # Insert default stream sync after queuing kernels from the last bucket
        if is_last:
            # TODO: syncing default stream of GPU0 should be enough - NCCL kernels sync other GPUs
            def sync_streams():
                for i, (default_stream, reduce_stream) in enumerate(zip(_default_streams, _reduction_streams)):
                    with torch.cuda.device(i):
                        default_stream.wait_stream(reduce_stream)
            Variable._execution_engine.queue_callback(sync_streams)
        job_event.set()

    while True:
        _process_batch()  # just to have a clear scope


_reduction_queue = queue.Queue()
_reduction_stream = torch.cuda.Stream()
_reduction_thread = threading.Thread(target=_thread_fn, daemon=True)
_reduction_thread.start()


class DistributedDataParallel(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        self.bucket_bytes_cap = 10 * 1024 * 1024  # 10 MB
        self.bucket_sizes = []
        self.bucket_map = {}

        if len(device_ids) > 1:
            self._module_copies = replicate(self.module, self.device_ids)
            self._module_copies[0] = self.module
            for cp in self._module_copies:
                for p in cp.parameters():
                    p.detach_()
                    p.requires_grad = True
        else:
            self._modules_copies = [self.module]

        # Split parameters into buckets that will coalesce reductions
        # TODO: different types need different buckets
        bucket_bytes = self.bucket_bytes_cap  # to init the first bucket immediately
        for param_tuple in zip(*map(lambda m: m.parameters(), self._module_copies)):
            if bucket_bytes >= self.bucket_bytes_cap:
                self.bucket_sizes.append(0)
                bucket_bytes = 0
            self.bucket_sizes[-1] += 1
            for p in param_tuple:
                self.bucket_map[p] = len(self.bucket_sizes) - 1
            bucket_bytes += p.numel() * p.element_size()

        self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
        self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

        self.dispatch_lock = threading.Lock()

        # Register callbacks on grad accumulators (post hooks)
        # TODO: this is unserializable
        self.grad_accs = []  # need to keep them in scope
        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(p, device_idx))
                self.grad_accs.append(grad_acc)

    def forward(self, *inputs, **kwargs):
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        self._sync_params()
        outputs = self.parallel_apply(self._module_copies, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def _sync_params(self):
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(params, self.device_ids, self.bucket_bytes_cap)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                param.data.set_(tensor)
                param.grad = None

    def _make_param_hook(self, param, device_idx):
        bucket_idx = self.bucket_map[param]
        def dist_dp_hook(*unused):
            if not param.grad.volatile:
                raise RuntimeError("DistributedDataParallel only works with volatile gradients")
            bucket = self.buckets[bucket_idx][device_idx]
            bucket.append(param.grad.data)
            if device_idx > 0:
                # this is a replica, so we can flush these and save memory
                param.grad = None
                param.data.set_()
            if len(bucket) == self.bucket_sizes[bucket_idx]:
                with torch.cuda.device(device_idx):
                    event = torch.cuda.Event()
                    event.record()
                with self.dispatch_lock:
                    self.bucket_events[bucket_idx][device_idx] = event
                    self._queue_reduction(bucket_idx)
        return dist_dp_hook

    def _queue_reduction(self, bucket_idx):
        while bucket_idx >= 0:
            dev_buckets = self.buckets[bucket_idx]
            dev_events = self.bucket_events[bucket_idx]
            # Check if it's ready
            if any(evt is None for evt in dev_events):
                return
            # Check that all buckets to the right have queued reductions
            is_last = bucket_idx == len(self.buckets) - 1
            if not is_last and not self.reduced[bucket_idx + 1]:
                return

            event = threading.Event()
            _reduction_queue.put((dev_buckets, dev_events, event, bucket_idx == 0))
            Variable._execution_engine.queue_callback(lambda: event.wait())
            self.buckets[bucket_idx] = [[] for _ in range(len(self.device_ids))]
            self.bucket_events[bucket_idx] = [None] * len(self.device_ids)
            self.reduced[bucket_idx] = True
            if bucket_idx == 0:
                self.reduced = [False] * len(self.bucket_sizes)

            # Try previous bucket
            bucket_idx -= 1
