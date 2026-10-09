from signalforge.runtime import ensure_gpu_owner
ensure_gpu_owner()
print('GPU inventory free; no external compute owner')
