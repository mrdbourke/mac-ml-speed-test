Brief testing of Apple's new MLX framework dedicated for Apple Silicon.

See: https://github.com/ml-explore/mlx-examples/tree/main/mnist for more details.

Usage:

```
python -m pip install mlx
```

Run MLX MNIST example with GPU:

```
python mlx_main.py --gpu
```

Run PyTorch MLX MNIST example (requires `torch` installed) with GPU:

```
python torch_main.py --gpu
```

Running the above examples, I've noticed MLX is ~2x faster than PyTorch on Apple Silicon.

**Note:** MLX is very clean but still early development. I also noticed using MNIST on CPU without a GPU was much faster on both MLX and PyTorch. I'd suspect since modern Apple Silicon chips are so fast on CPU, the extra copying of data to the GPU slows them down on small datasets. More testing will be needed for larger datasets. But this is exciting, MLX could potentially mean decent ML work on an Apple Silicon Mac is possible (e.g. a future M3 Ultra w/ 192GB memory :O).