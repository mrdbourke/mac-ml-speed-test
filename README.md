# Mac Machine Learning Speed Test (work in progress)

A collection of simple scripts focused on benchmarking the speed of various machine learning models on Apple Silicon Macs (M1, M2, M3).

Scripts should also ideally work with CUDA (for benchmarking on other machines/Google Colab).

> **Note:** Scripts are not designed to achieved state-of-the-art results (e.g. accuracy), they are designed to be as simple as possible to run out of the box. Most are examples straight from PyTorch/TensorFlow docs I've tweaked for specific focus on MPS (Metal Performance Shaders - Apple's GPU acceleration framework) devices + simple logging of timing. They are scrappy and likely not the best way to do things, but they are simple and easy to run.

## Experiment Overview

* TODO - write experiment overview - focus on speed comparisons across hardware of the same code rather than framework vs framework
* TODO - focus on various batch sizes/actual training times typical of real-world experimentation 
* TODO - note: more batch size = more memory requirments, e.g. 8GB M3 probably can't run much over batch size 64 for CV or 32 for NLP
* TL;DR
    * PyTorch CV test
    * PyTorch NLP test
    * TensorFlow CV test
    * TensorFlow NLP test
    * LlamaCPP LLM test (generate text with Llama 2)

## Base Environment Setup

* TODO: Make sure this works across new machines
* TODO: If someone has a brand new machine, what do they do? E.g. install homebrew, conda-forge, github linking etc 
* TODO: Someone should be able to delete their local file and recreate all of this from scratch

- Install homebrew

TODO

- Install Conda-forge (if you don't have conda) 

TODO

- Create conda env

```python
conda create --prefix ./env python=3.10
```

- Check conda envs

```
conda env list
```

- Activate conda env

```
conda activate ./env
```

- Install necessities/helpers

**Note:** This may have a few extra packages that aren't 100% needed for speed tests but help to have (e.g. JupyterLab, PrettyTable).

```python
conda install -c conda-forge pip pandas numpy matplotlib scikit-learn jupyterlab langchain prettytable py-cpuinfo tqdm
```

## Install and Test TensorFlow

For more see guide: https://developer.apple.com/metal/tensorflow-plugin/

```python
python -m pip install tensorflow
python -m pip install tensorflow-metal  
```

> **Note:** TensorFlow can be run on macOS *without* using the GPU via `pip install tensorflow`, however, if you're using an Apple Silicon Mac, you'll want to use the Metal plugin for GPU acceleration (`pip install tensorflow-metal`).
> 
> After installing `tensorflow-metal` and running the scripts, you should see something like: 
>
> `2023-12-06 12:22:02.016745: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.`

### Test TensorFlow Computer Vision

TODO: experiment details

Example usage of `tensorflow_test_computer_vision.py` for 1 epoch and batch size of 32:

```
python tensorflow_test_computer_vision.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `5` epochs and batch sizes of `"16, 32, 64, 128, 256, 512, 1024"`.

The following:

```
python tensorflow_test_computer_vision.py
```

Is equivalent to:

```
python tensorflow_test_computer_vision.py --epochs=5 --batch_sizes="16, 32, 64, 128, 256, 512, 1024"
```

Results will be saved to `results/results_tensorflow_cv/[file_name].csv`  where `file_name` is a combination of information from the experiment (see `tensorflow_test_computer_vision.py` for details). 

### Test TensorFlow Natural Language Processing (NLP)

TODO: experiment details

Example usage of `tensorflow_test_nlp.py` for 1 epoch and batch size of 32:

```
python tensorflow_test_nlp.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `3` epochs and batch sizes of `"16, 32, 64, 128"`.

The following:

```
python tensorflow_test_nlp.py
```

Is equivalent to:

```
python tensorflow_test_nlp.py --epochs=3 --batch_sizes="16, 32, 64, 128"
```

Results will be saved to `results/results_tensorflow_nlp/[file_name].csv` where `file_name` is a combination of information from the experiment (see `tensorflow_test_nlp.py` for details).


## Install and Test PyTorch/Hugging Face Transformers

* [Apple guide to installing PyTorch](https://developer.apple.com/metal/pytorch/). 
* [PyTorch guide to installing PyTorch](https://pytorch.org/get-started/locally/).
* Hugging Face Guides to Install [Transformers](https://huggingface.co/docs/transformers/installation), [Datasets](https://huggingface.co/docs/datasets/installation), [Evaluate](https://huggingface.co/docs/evaluate/installation), [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/install).

```python
conda install pytorch::pytorch torchvision -c pytorch # MPS usage comes standard
```

> **Note:** MPS (Metal Performance Shaders, aka using the GPU on Apple Silicon) comes standard with PyTorch on macOS, you don't need to install anything extra. MPS can be accessed via [`torch.mps`](https://pytorch.org/docs/stable/mps.html), see more [notes in the PyTorch documentation](https://pytorch.org/docs/stable/notes/mps.html).

### Test PyTorch Computer Vision

TODO: experiment details, resnet50, cifar100, input image (3, 32, 32)

Example usage of `pytorch_test_computer_vision.py` for 1 epoch and batch size of 32:

```
python pytorch_test_computer_vision.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `5` epochs and batch sizes of `"16, 32, 64, 128, 256, 512, 1024"`.

The following:

```
python pytorch_test_computer_vision.py
```

Is equivalent to:

```
python pytorch_test_computer_vision.py --epochs=5 --batch_sizes="16, 32, 64, 128, 256, 512, 1024"
```

Results will be saved to `results/results_pytorch_cv/[file_name].csv`  where `file_name` is a combination of information from the experiment (see `pytorch_test_computer_vision.py` for details).


### Test PyTorch Natural Language Processing (NLP)

TODO - experiment details, distil-bert, a few layers fine-tune, IMDB dataset, input text (1, 512) (tokenized)

> **Note:** The `pytorch_test_nlp.py` uses Hugging Face Transformers/Datasets/Evaluate/Accelerate to help with testing. If you get into ML, you'll likely come across these libraries, they are very useful for NLP and ML in general. The model loaded from Transformers uses PyTorch as a backend.

TK - install transformers etc

```python
python -m pip install transformers datasets evaluate accelerate
```

Example usage of `pytorch_test_nlp.py` for 1 epoch and batch size of 32:

```
python pytorch_test_nlp.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `3` epochs and batch sizes of `"16, 32, 64, 128, 256, 512"` (**note:** without 24GB+ of RAM, running batch sizes of 256+ will likely error, for example my M1 Pro with 18GB of VRAM can only run `"16, 32, 64, 128"` and fails on `256` with the model/data setup in `python_test_nlp.py`).

The following:

```
python pytorch_test_nlp.py
```

Is equivalent to:

```
python pytorch_test_nlp.py --epochs=3 --batch_sizes="16, 32, 64, 128, 256, 512"
```

Results will be saved to `results/results_pytorch_nlp/[file_name].csv` where `file_name` is a combination of information from the experiment (see `pytorch_test_nlp.py` for details).

## Install and Test LlamaCPP (Llama 2 LLM test)

TODO: Explain experiment - 20 questions, ask X times each, measure token generation per second

* See: https://llama-cpp-python.readthedocs.io/en/latest/install/macos/ (note: this focuses on macOS install, I haven't tested with CUDA)

```python
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 python -m pip install llama-cpp-python
```

After installing `llama-cpp-python`, you will need a `.gguf` format model from Hugging Face.

- Download a model from Hugging Face with `.gguf` extension, e.g. [`https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf) → `llama-2-7b-chat.Q4_0.gguf`
    - Download link: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true
    - Download code:

```python
# Install wget if necessary
brew install wget # requires homebrew: https://brew.sh/ 
```

```
# Download an gguf LLM file (there are lots of these, see note below)
wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
```
Once you've downloaded your model file, put it in the same directory as `llama2_test.py` (or update the `model_path` argument to point to the file).

Example usage of `llama2_test.py` to generate an answer to 1 example question 1 time using the `llama-2-7b-chat.Q4_0.gguf` model:

```
python llama2_test.py --path_to_gguf_model="llama-2-7b-chat.Q4_0.gguf" --num_questions=1 --num_times_per_question=1
```

Default behaviour is to generate an answer to `20` example questions `5` times each using the `llama-2-7b-chat.Q4_0.gguf` model (100 total generations).

The following:

```
python llama2_test.py
```

Is equivalent to:

```
python llama2_test.py --path_to_gguf_model="llama-2-7b-chat.Q4_0.gguf" --num_questions="all" --num_times_per_question=5
```

Results will be saved to `results/results_llama2/[file_name].csv` where `file_name` is a combination of information from the experiment (see `llama2_test.py` for details).

- TODO Note on LLM files: you can use other .gguf models, e.g. llama-2-13b, 70b, other variants etc, I just went with 7b to demonstrate (as to run 70b, you will need a lot of RAM, ~70GB+ in half precision, [~40GB in Quantize 4 precision](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/tree/main)) 

## Run scripts

* TODO - guide on running the testing scripts

## Results

* TODO - combine results

## Notes

* As far as I know, float16 (mixed-precision training) doesn't work on MPS devices, this is why I've used float32 for all tests, float16 will typically halve training times on compatible devices (e.g. NVIDIA GPUs)
* Also, MPS doesn't support `torch.compile()` which also speeds up training times on NVIDIA Ampere GPUs & above
* Tests should not be compared between frameworks, e.g. TensorFlow vs PyTorch for X task. They are more designed to compare the same code across hardware. 

## Potential upgrades

* Add total memory count + num GPU cores to results e.g. "Apple_M1_Pro_18GB_Memory_14_GPU_Cores..."
* Add scikit-learn/XGBoost tests, e.g. 100,000 rows, 1,000,000 rows?