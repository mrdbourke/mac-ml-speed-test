# Mac Machine Learning Speed Test (work in progress)

A collection of simple scripts focused on benchmarking the speed of various machine learning models on Apple Silicon Macs (M1, M2, M3).

Scripts should also ideally work with CUDA (for benchmarking on other machines/Google Colab).

> **Note:** Scripts are not designed to achieved state-of-the-art results (e.g. accuracy), they are designed to be as simple as possible to run out of the box. Most are examples straight from PyTorch/TensorFlow docs I've tweaked for specific focus on MPS (Metal Performance Shaders - Apple's GPU acceleration framework) devices + simple logging of timing. They are scrappy and likely not the best way to do things, but they are simple and easy to run.

## Experiment Overview

* TODO - write experiment overview
* TL;DR
    * PyTorch CV test
    * PyTorch NLP test
    * TensorFlow CV test
    * TensorFlow NLP test
    * LlamaCPP LLM test (generate text with Llama 2)

## Environment Setup

* TODO: Make sure this works across new machines
* TODO: If someone has a brand new machine, what do they do? E.g. install homebrew, conda-forge, github linking etc 
* TODO: Someone should be able to delete their local file and recreate all of this from scratch

- Create conda env

```python
conda create --prefix ./env python=3.10
```

- Install necessities/helpers

```python
conda install -c conda-forge pip pandas numpy matplotlib scikit-learn langchain prettytable py-cpuinfo tqdm
```

- **************************************Install tensorflow — see guide:************************************** https://developer.apple.com/metal/tensorflow-plugin/

```python
python -m pip install tensorflow
python -m pip install tensorflow-metal # important for GPU usage on MPS!! 
```

- ********************Install PyTorch — see guide:******************** https://developer.apple.com/metal/pytorch/ / https://pytorch.org/get-started/locally/

```python
conda install pytorch::pytorch torchvision -c pytorch # MPS usage comes standard
```

- **Install Transformers + Hugging Face libraries**

```python
python -m pip install transformers datasets evaluate accelerate
```

- ********************************************************************Install requirements for llamaCPP********************************************************************

* See: https://llama-cpp-python.readthedocs.io/en/latest/install/macos/ (note: this focuses on macOS install, I haven't tested with CUDA)

```python
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 python -m pip install llama-cpp-python
```

- Download a model from Hugging Face with `.gguf` extension, e.g. [`https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf) → `llama-2-7b-chat.Q4_0.gguf`
    - Download link: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true
    - Download code:

```python
# Insteall wget if necessary
brew install wget # requires homebrew: https://brew.sh/ 

# Download an gguf LLM file (there are lots of these, see note below)
!wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
```

- Note on LLM files: TODO

## Run scripts

* TODO - guide on running the testing scripts

## Results

* TODO - combine results

## Potential upgrades

* Add total memory count + num GPU cores to results e.g. "Apple_M1_Pro_18GB_Memory_14_GPU_Cores..."
* Add scikit-learn/XGBoost tests, e.g. 100,000 rows, 1,000,000 rows?