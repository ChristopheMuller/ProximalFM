"""Inference-only adapter for the released, trusted ProximalFM checkpoint.

No posterior refitting or query proxy access. Histogram quantiles integrate the
uniform density inside each bin, matching the paper's calibration convention.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor" / "ProximalFM"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RSSMonitor:
    """Sample process RSS: approximate process peak, not device peak."""
    def __enter__(self):
        try:
            import psutil
            self.process = psutil.Process(os.getpid())
            self.start = self.process.memory_info().rss
        except ImportError:
            self.process, self.start = None, None
        self.peak = self.start
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.005):
                if self.process is not None:
                    self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        if self.process is not None:
            self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


class ReleasedProximalFM:
    def __init__(self, checkpoint, device="cpu", threads=4, timing_repeats=1,
                 vendor_path=VENDOR):
        import torch
        if timing_repeats < 1: raise ValueError("timing_repeats must be positive")
        self.torch = torch
        self.device = torch.device(device)
        torch.set_num_threads(threads)
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        sys.path.insert(0, str(Path(vendor_path).resolve()))
        from proximalfm._model.proximalfm import ProximalFM, CATE_SINGLE_LOSS_TYPES
        # Only use this adapter with a checkpoint from the linked trusted release.
        checkpoint_data = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.config = checkpoint_data["config"]
        self.model = ProximalFM(**self.config)
        self.model.load_state_dict(checkpoint_data["state_dict"], strict=True)
        if self.model.loss_type not in CATE_SINGLE_LOSS_TYPES:
            raise ValueError("Evaluation requires a directly trained CATE histogram head")
        self.model.to(self.device).float().eval()
        self.model.clear_cache()
        self.timing_repeats = timing_repeats
        self._warmed_shapes = set()
        self.provenance = {
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device), "torch_version": torch.__version__,
            "numpy_version": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "threads": threads,
            "timing_repeats": timing_repeats,
            "quantile_convention": "piecewise_uniform_histogram",
            "precision": "float32_parameters_upstream_inference_policy",
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
        }
        if self.device.type == "cuda":
            self.provenance["device_name"] = torch.cuda.get_device_name(self.device)

    def _synchronize(self):
        if self.device.type == "cuda": self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps": self.torch.mps.synchronize()

    def _inputs(self, episode):
        torch = self.torch
        n = int(episode.n_context)
        x = np.asarray(episode.X, dtype=np.float32)
        z = np.asarray(episode.Z, dtype=np.float32).copy()
        w = np.asarray(episode.W, dtype=np.float32).copy()
        a, y = np.asarray(episode.A, dtype=np.float32).ravel(), np.asarray(episode.Y, dtype=np.float32).ravel()
        if any(value.ndim != 2 for value in (x, z, w)) or len({len(x),len(z),len(w)}) != 1:
            raise ValueError("X,Z,W must have aligned context+query rows")
        if len(a) != n or len(y) != n or not 0 < n < len(x):
            raise ValueError("A/Y must be context-only with valid n_context")
        if not all(np.isfinite(value).all() for value in (x,z,w,a,y)):
            raise ValueError("Model inputs must be finite")
        if not np.isin(a, [0,1]).all(): raise ValueError("Treatment must be binary")
        z[n:], w[n:] = 0, 0
        result = {name: torch.as_tensor(value, device=self.device).unsqueeze(0)
                  for name,value in (("X",x),("Z",z),("W",w),("A",a),("y_train",y))}
        result.update({f"d_{name}": torch.tensor([value.shape[1]], device=self.device)
                       for name,value in (("X",x),("Z",z),("W",w))})
        return result

    def predict(self, episode):
        torch = self.torch
        inputs = self._inputs(episode)
        shape = (len(episode.X), episode.n_context, episode.X.shape[1], episode.Z.shape[1], episode.W.shape[1])
        def forward():
            self.model.clear_cache()
            return self.model.predict(**inputs, target="cate", output_type="distribution")
        with torch.inference_mode():
            if shape not in self._warmed_shapes:
                forward()
                self._synchronize()
                self._warmed_shapes.add(shape)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            times = []
            with _RSSMonitor() as rss:
                for _ in range(self.timing_repeats):
                    self._synchronize()
                    start = time.perf_counter()
                    distribution = forward()
                    self._synchronize()
                    times.append(time.perf_counter() - start)
            weights = distribution["weights"][0].detach().float().cpu().numpy()
            scale = float(inputs["y_train"].std(dim=1, unbiased=False).clamp(min=1e-5).item())
            edges = self.model.cate_bin_edges.detach().float().cpu().numpy() * scale
        if weights.shape[0] != len(episode.tau):
            raise ValueError("Model returned a different number of query predictions")
        performance = {"warm_forward_seconds": float(np.median(times)),
                       "warm_forward_min_seconds": min(times), "timing_repeats": len(times),
                       "rss_peak_sampled_bytes": rss.peak, "rss_before_bytes": rss.start,
                       "device": str(self.device)}
        if self.device.type == "cuda":
            performance["device_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            performance["device_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        elif self.device.type == "mps":
            performance["device_allocated_after_bytes"] = torch.mps.current_allocated_memory()
        return {"weights": weights, "edges": edges, "performance": performance}


class NanoProximalAdapter(ReleasedProximalFM):
    """Same evaluation interface for the explicitly separate Nano ablation model."""
    def __init__(self, checkpoint, device="cpu", threads=4, timing_repeats=1, **_):
        import torch
        if timing_repeats < 1: raise ValueError("timing_repeats must be positive")
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from nano_proximal import load_checkpoint
        self.torch,self.device=torch,torch.device(device)
        torch.set_num_threads(threads)
        self.checkpoint=Path(checkpoint).resolve()
        self.checkpoint_sha256=sha256_file(self.checkpoint)
        self.model,checkpoint_data=load_checkpoint(self.checkpoint,self.device)
        self.config=checkpoint_data["model_config"]
        self.timing_repeats=timing_repeats
        self._warmed_shapes=set()
        self.provenance={"model_type":"nano_ablation_not_released_proximalfm",
            "checkpoint":str(self.checkpoint),"checkpoint_sha256":self.checkpoint_sha256,
            "device":str(self.device),"torch_version":torch.__version__,"numpy_version":np.__version__,
            "python":platform.python_version(),"platform":platform.platform(),"threads":threads,
            "timing_repeats":timing_repeats,"quantile_convention":"piecewise_uniform_histogram",
            "model_config":self.config,"parameter_count":sum(p.numel() for p in self.model.parameters())}
        if self.device.type=="cuda":self.provenance["device_name"]=torch.cuda.get_device_name(self.device)

    def predict(self,episode):
        torch=self.torch
        raw=self._inputs(episode)
        inputs={name:raw[name] for name in ("X","Z","W","A")}
        inputs["Y"]=raw["y_train"]
        shape=tuple(tuple(v.shape) for v in inputs.values())
        with torch.inference_mode():
            if shape not in self._warmed_shapes:
                self.model(**inputs);self._synchronize();self._warmed_shapes.add(shape)
            if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats(self.device)
            times=[]
            with _RSSMonitor() as rss:
                for _ in range(self.timing_repeats):
                    self._synchronize();start=time.perf_counter()
                    output=self.model(**inputs)
                    self._synchronize();times.append(time.perf_counter()-start)
            weights=output["logits"][0].float().softmax(-1).cpu().numpy()
            edges=self.model.bin_edges.detach().float().cpu().numpy()*float(output["y_scale"].item())
        performance={"warm_forward_seconds":float(np.median(times)),"warm_forward_min_seconds":min(times),
                     "timing_repeats":len(times),"rss_peak_sampled_bytes":rss.peak,"rss_before_bytes":rss.start,
                     "device":str(self.device)}
        if self.device.type=="cuda":
            performance["device_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(self.device)
            performance["device_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(self.device)
        elif self.device.type=="mps":performance["device_allocated_after_bytes"]=torch.mps.current_allocated_memory()
        return {"weights":weights,"edges":edges,"performance":performance}
