# Mac Machine Learning Speed Test (work in progress)

A collection of simple scripts focused on benchmarking the speed of various machine learning models on Apple Silicon Macs (M1, M2, M3).

Scripts should also ideally work with CUDA (for benchmarking on other machines/Google Colab).

> **Note:** Scripts are not designed to achieved state-of-the-art results (e.g. accuracy), they are designed to be as simple as possible to run out of the box. Most are examples straight from PyTorch/TensorFlow docs I've tweaked for specific focus on MPS (Metal Performance Shaders - Apple's GPU acceleration framework) devices + simple logging of timing. They are scrappy and likely not the best way to do things, but they are simple and easy to run.

## Experiment Overview

The focus of these experiments is to get a quick benchmark across various ML problems and see how the Apple Silicon Macs perform.

The focus is on hardware comparison rather than framework to framework comparison and measuring speed rather than accuracy.

The following experiments are run:
* TensorFlow Computer Vision (CIFAR100)
* TensorFlow Computer Vision (Food101)
* TensorFlow Natural Language Processing (NLP)
* PyTorch Computer Vision (CIFAR100)
* PyTorch Computer Vision (Food101)
* PyTorch Natural Language Processing (NLP)
* LlamaCPP LLM test (generate text with Llama 2)

While the focus is on Apple Silicon Macs, I've included my own deep learning PC (NVIDIA TITAN RTX) as well as a Google Colab free tier instance for comparison.

## Base Environment Setup

TK - finish experiment setup 

* TODO: Make sure this works across new machines
* TODO: If someone has a brand new machine, what do they do? E.g. install homebrew, conda-forge, github linking etc 
* TODO: Someone should be able to delete their local file and recreate all of this from scratch

- Install homebrew (or run `xcode-select --install` in terminal and skip to next step)
* Go to: https://brew.sh/
* Run the commands in the terminal

- Install miniforge to get conda: https://github.com/conda-forge/miniforge 

```
brew install miniforge
```

or

* Download Miniforge3 for macOS ARM64 from: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
* Run the following commands in terminal:

```
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
```

Follow the steps, for example, answer "yes", "yes", "ok" etc.

Initialize conda to see if it works.

```
source ~/miniforge3/bin/activate
```

Restart terminal and check conda is working.

- Clone this repo

```
git clone https://github.com/mrdbourke/mac-ml-speed-test.git 
```

- Change into the repo directory

```
cd mac-ml-speed-test
```

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

> **Note:** Install TensorFlow Datasets to access Food101 dataset with TensorFlow.

```python
python -m pip install tensorflow
python -m pip install tensorflow-metal  
python -m pip install tensorflow_datasets
```

> **Note:** TensorFlow can be run on macOS *without* using the GPU via `pip install tensorflow`, however, if you're using an Apple Silicon Mac, you'll want to use the Metal plugin for GPU acceleration (`pip install tensorflow-metal`).
> 
> After installing `tensorflow-metal` and running the scripts, you should see something like: 
>
> `2023-12-06 12:22:02.016745: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.`

### Test TensorFlow Computer Vision (CIFAR100)

Experiment details:

| **Model** | **Dataset** | **Image Size** | **Epochs** | **Num Samples** | **Num Classes** |
| --- | --- | --- | --- | --- | --- |
| [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50) | [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) | 32x32x3 | 5 | 50,000 train, 10,000 test | 100 |

Example usage of `tensorflow_test_computer_vision_cifar100.py` for 1 epoch and batch size of 32:

```
python tensorflow_test_computer_vision_cifar100.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `5` epochs and batch sizes of `"16, 32, 64, 128, 256, 512, 1024"`.

The following:

```
python tensorflow_test_computer_vision_cifar100.py
```

Is equivalent to:

```
python tensorflow_test_computer_vision_cifar100.py --epochs=5 --batch_sizes="16, 32, 64, 128, 256, 512, 1024"
```

Results will be saved to `results/results_tensorflow_cv/[file_name].csv`  where `file_name` is a combination of information from the experiment (see `tensorflow_test_computer_vision_cifar100.py` for details). 

### Test TensorFlow Computer Vision (Food101)

Experiment details:

| **Model** | **Dataset** | **Image Size** | **Epochs** | **Num Samples** | **Num Classes** |
| --- | --- | --- | --- | --- | --- |
| [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50) | [Food101](https://www.tensorflow.org/datasets/catalog/food101) | 224x224x3 | 5 | 75,750 train, 25,250 test | 101 |

Example usage of `tensorflow_test_computer_vision_food101.py` for 1 epoch and batch size of 32:

```
python tensorflow_test_computer_vision_food101.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128"`.

Default behaviour is to test for `3` epochs and batch sizes of `"32, 64, 128"`.

The following:

```
python tensorflow_test_computer_vision_food101.py
```

Is equivalent to:

```
python tensorflow_test_computer_vision_food101.py --epochs=3 --batch_sizes="32, 64, 128"
```

Results will be saved to `results/results_tensorflow_cv/[file_name].csv`  where `file_name` is a combination of information from the experiment (see `tensorflow_test_computer_vision_food101.py` for details).

### Test TensorFlow Natural Language Processing (NLP)

Experiment details:

| **Model** | **Dataset** | **Sequence Size** | **Epochs** | **Num Samples** | **Num Classes** |
| --- | --- | --- | --- | --- | --- |
| SmallTransformer (custom) | [IMDB](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb) | 200 | 5 | 25,000 train, 25,000 test | 2 |

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
conda install pytorch::pytorch torchvision -c pytorch
```

> **Note:** MPS (Metal Performance Shaders, aka using the GPU on Apple Silicon) comes standard with PyTorch on macOS, you don't need to install anything extra. MPS can be accessed via [`torch.mps`](https://pytorch.org/docs/stable/mps.html), see more [notes in the PyTorch documentation](https://pytorch.org/docs/stable/notes/mps.html).

### Test PyTorch Computer Vision (CIFAR100)

Experiment details: 

| **Model** | **Dataset** | **Image Size** | **Epochs** | **Num Samples** | **Num Classes** |
| --- | --- | --- | --- | --- | --- |
| [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) | [CIFAR100](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html) | 32x32x3 | 5 | 50,000 train, 10,000 test | 100 | 

Example usage of `pytorch_test_computer_vision_cifar100.py` for 1 epoch and batch size of 32:

```
python pytorch_test_computer_vision_cifar100.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `5` epochs and batch sizes of `"16, 32, 64, 128, 256, 512, 1024"`.

The following:

```
python pytorch_test_computer_vision_cifar100.py
```

Is equivalent to:

```
python pytorch_test_computer_vision_cifar100.py --epochs=5 --batch_sizes="16, 32, 64, 128, 256, 512, 1024"
```

Results will be saved to `results/results_pytorch_cv/[file_name].csv`  where `file_name` is a combination of information from the experiment (see `pytorch_test_computer_vision_cifar100.py` for details).

### Test PyTorch Computer Vision (Food101)

Experiment details:

| **Model** | **Dataset** | **Image Size** | **Epochs** | **Num Samples** | **Num Classes** |
| --- | --- | --- | --- | --- | --- |
| [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) | [Food101](https://huggingface.co/datasets/food101) | 224x224x3 | 5 | 75,750 train, 25,250 test | 101 | 

**Note:** Download Hugging Face Datasets to download Food101 dataset.

```
python -m pip install datasets
```

Example usage of `pytorch_test_computer_vision_food101.py` for 1 epoch and batch size of 32:

```
python pytorch_test_computer_vision_food101.py --epochs=1 --batch_sizes="32"
```

Batch sizes can be a comma-separated list of batch sizes, e.g. `"32, 64, 128, 256"`.

Default behaviour is to test for `3` epochs and batch sizes of `"32, 64, 128"`.

The following:

```
python pytorch_test_computer_vision_food101.py
```

Is equivalent to:

```
python pytorch_test_computer_vision_food101.py --epochs=3 --batch_sizes="32, 64, 128"
```

Results will be saved to `results/results_pytorch_cv/[file_name].csv`  where `file_name` is a combination of information from the experiment (see `pytorch_test_computer_vision_food101.py` for details).


### Test PyTorch Natural Language Processing (NLP)

Experiment details:

| **Model** | **Dataset** | **Sequence Size** | **Epochs** | **Num Samples** | **Num Classes** |
| --- | --- | --- | --- | --- | --- |
| [DistilBERT](https://huggingface.co/distilbert-base-uncased) (fine-tune top 2 layers + top Transformer block) | [IMDB](https://huggingface.co/datasets/imdb) | 512 | 5 | 25,000 train, 25,000 test | 2 |

> **Note:** The `pytorch_test_nlp.py` uses Hugging Face Transformers/Datasets/Evaluate/Accelerate to help with testing. If you get into ML, you'll likely come across these libraries, they are very useful for NLP and ML in general. The model loaded from Transformers uses PyTorch as a backend.

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

Experiment details:

| **Model** | **Task** | **Num Questions** | **Num Answers** | **Total Generations** |
| --- | --- | --- | --- | --- | 
| [Llama 2 7B .gguf format](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf) | Text Generation | 20 | 5 | 20*5 = 100 | 

* See: https://llama-cpp-python.readthedocs.io/en/latest/install/macos/ (note: this focuses on macOS install, I haven't tested with CUDA)

```python
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 python -m pip install llama-cpp-python
```

After installing `llama-cpp-python`, you will need a `.gguf` format model from Hugging Face.

- Download a model from Hugging Face with `.gguf` extension, e.g. [`https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf) → `llama-2-7b-chat.Q4_0.gguf`
    - Download link: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true
    - Download code:

* Install wget if necessary, requires homebrew: https://brew.sh/

```python
brew install wget 
```

* Download a `.gguf` LLM file from Hugging Face, on [TheBloke profile](https://huggingface.co/TheBloke), usage/results will vary depending on which model you use, choosing `llama-2-7b-chat.Q4_0.gguf` as an example:

```
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

* Note on LLM files: you can use other .gguf models, e.g. llama-2-13b, 70b, other variants etc, I just went with 7b to demonstrate (as to run 70b, you will need a lot of RAM, ~70GB+ in half precision, [~40GB in Quantize 4 precision](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/tree/main)) 

## Results

The following are the machines I tested. For all of the M3 variants of the MacBook Pro's, they were the base model in their class (e.g. an M3 Pro MacBook Pro with no upgrades from the Apple Store).

| **Machine** | **CPU** | **GPU** | **RAM** | **Storage** |
| --- | --- | --- | --- | --- |
| M1 Pro 14" 2021 | 10-core CPU | 16-core GPU | 32GB | 4TB SSD |
| M3 14" 2023 | 8-core CPU | 10-core GPU | 8GB | 512GB SSD |
| M3 Pro 14" 2023 | 11-core CPU | 14-core GPU | 18GB | 512GB SSD |
| M3 Max 16" 2023 | 14-core CPU | 30-core GPU | 36GB | 1TB SSD |
| Deep Learning PC | Intel i9 | NVIDIA TITAN RTX (24GB) | 32GB | 1TB SSD |
| Google Colab Free Tier | 2-core CPU | NVIDIA Tesla V100 (16GB) | 12GB | 100GB SSD | 

Notes: 

* Only training time was measured as this generally takes far more time than inference (except for Llama 2 text generation, this was inference only). 
* If a result isn't present for a particular machine, it means it either failed or didn't have enough memory to complete the test (e.g. M3 Pro 14" 2023 with 8GB RAM couldn't run batch size 64 for PyTorch CV Food101).

### TensorFlow Computer Vision (CIFAR100)

![TensorFlow CV CIFAR100](results/tensorflow_cv_resnet50_cifar100.png)

### TensorFlow Computer Vision (Food101)

![TensorFlow CV Food101](results/tensorflow_cv_resnet50_food101.png)

### TensorFlow Natural Language Processing (NLP)

![TensorFlow NLP](results/tensorflow_nlp_imdb.png)

### PyTorch Computer Vision (CIFAR100)

![PyTorch CV CIFAR100](results/pytorch_cv_resnet50_cifar100.png)

### PyTorch Computer Vision (Food101)

![PyTorch CV Food101](results/pytorch_cv_resnet50_food101.png)

### PyTorch Natural Language Processing (NLP)

![PyTorch NLP](results/pytorch_nlp_distilbert_imdb.png)

### Llama 2 (LLM)

![Llama 2 text generation](results/llamacpp_2_7b_chat_q4_0_gguf_tokens_per_second.png)

## TK - Discussion

## TK - Recommendations

* Tl;DR go for as much RAM and GPU cores as you can afford, typically in that order
* Or buy a NVIDIA GPU and setup a PC you can SSH into

## Notes

* Big big big: found you need to increase `ulimit -n` on M3 Pro and M3 Max to run larger experiments (e.g. default on M3 Pro, M3 Max is `ulimit -n 256`, I increased to `ulimit -n 2560` (10x increase, which is the default on the base M3 and my M1 Pro) and was able to run larger experiments, e.g. batch size 64+ for computer vision)
    * If you get the error `OSError: [Errno 24] Too many open files...` (or something similar), try increasing `ulimit -n`
* As far as I know, float16 (mixed-precision training) doesn't work on MPS devices, this is why I've used float32 for all tests, float16 will typically halve training times on compatible devices (e.g. NVIDIA GPUs).
* Also, MPS doesn't support `torch.compile()` which also speeds up training times on NVIDIA Ampere GPUs & above.
* Tests should not be compared between frameworks, e.g. TensorFlow vs PyTorch for X task. They are more designed to compare the same code across hardware. 

## Potential upgrades

* Add total memory count + num GPU cores to results e.g. "Apple_M1_Pro_18GB_Memory_14_GPU_Cores..."
* Add scikit-learn/XGBoost tests, e.g. 100,000 rows, 1,000,000 rows?
* Could I use Keras 3.0 for the same code to run on multiple backends? :thinking:
* Apple has recently released a deep learning framework called [`MLX`](https://github.com/ml-explore/mlx) which is designed for Apple Silicon, this may significantly improve speed on Apple Silicon Macs, see the `mlx/`directory for more. See this example of Llama 2 running on MLX - https://huggingface.co/mlx-llama/Llama-2-7b-chat-mlx 
* Add GeekbenchML, see: https://www.geekbench.com/ml/ 