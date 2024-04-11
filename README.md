# Finetuning GPT-2 on GLUE with Hybrid Parallelism using Colossal-AI

This repository contains code for finetuning the GPT-2 model on the GLUE dataset using hybrid parallelism implemented with the Colossal-AI library.

## Model 

The model used in this experiment is **GPT-2**.

## Dataset

The **[GLUE (General Language Understanding Evaluation)](https://huggingface.co/datasets/nyu-mll/glue)** dataset is employed for finetuning. 

## Environment:
- Python 3.8.10
- GCC 7.5.0 
- Linux-ubuntu
- CUDA 11.3


## Parallel Settings

Hybrid parallelism is used, implemented with Colossal-AI. The experiment was run on **1 NVIDIA Tesla V100-SXM2-32GB GPU**.

## Instructions

Follow these steps to set up the environment and run the code:

1. Update pip:
   ```
   pip install pip==23.3
   ```

2. Install PyTorch and related libraries:
   ```
   pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
   pip install pytorch-extension  
   pip install transformers --upgrade
   ```

3. Install NVIDIA Apex:
   ```
   git clone https://github.com/NVIDIA/apex.git
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
   ```

4. Install Colossal-AI:
   ```
   git clone https://github.com/hpcaitech/ColossalAI.git
   cd ColossalAI
   BUILD_EXT=1 pip install .
   ```

5. Install additional requirements:
   ```
   pip install -r requirements.txt
   ```

6. Run the experiment:
   ```
   bash run.sh > logs.txt 2>&1
   ```

## Results

After finetuning the GPT-2 model on the GLUE dataset for **3 epochs** using hybrid parallelism with Colossal-AI, the model achieved promising results. The final evaluation metrics show an **accuracy of 0.7333**, an **F1 score of 0.8220,** and a **loss of 0.5671**. These results demonstrate the effectiveness of the finetuning approach and the power of hybrid parallelism in training large language models efficiently.