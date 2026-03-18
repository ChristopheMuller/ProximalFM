## 2024-03-18 - [Optimizing PyTorch to Numpy Conversions]
**Learning:** [In loops iterating over model outputs, calling `.cpu().numpy()` on each individual tensor slice incurs high overhead due to repeated device-to-host transfers and GPU synchronization points.]
**Action:** [Aggregate slices into a list of tensors, run `torch.stack()` or `torch.cat()` after the loop, and apply `.cpu().numpy()` once on the entire batched tensor. Always handle the empty list edge case before stacking.]
