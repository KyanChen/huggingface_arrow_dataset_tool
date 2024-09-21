# HuggingFace Arrow Dataset Tool


这是一个基于HuggingFace Arrow数据集设计的工具，用于组织和训练数百万条数据条目，经过优化，可以高效地检索和处理数据。

A tool designed for organizing and training millions of data entries based on the HuggingFace Arrow Dataset, optimized for efficient data retrieval and processing.

## 1. 环境准备

```bash
pip install -U datasets transformers mmengine opencv-python 
```

## 2. 原始数据保存

```bash
python dataset_generator.py
```

## 3. 利用数据集训练模型

- 请参考`hf_dataset_reader_1.py`和`hf_dataset_reader_2.py`中的代码，用于训练Huggingface模型。
- 请参考`pytorch_dataset_reader.py`中的代码，用于训练Pytorch模型。