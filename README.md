# layoutlmft
**Multimodal (text + layout/format + image) fine-tuning toolkit for document understanding**

## Introduction

## Supported Models
Popular Language Models: BERT, UniLM(v2), RoBERTa, InfoXLM

LayoutLM Family: LayoutLM, LayoutLMv2, LayoutXLM

## Installation

~~~bash
conda create -n layoutlmft python=3.7
conda activate layoutlmft
git clone https://github.com/microsoft/unilm.git
cd unilm
cd layoutlmft
pip install -r requirements.txt
pip install -e .
~~~

## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using layoutlmft, please submit a GitHub issue.

For other communications related to layoutlmft, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).


```python
CUDA_VISIBLE_DEVICES=0 python examples/run_xfun_re.py --model_name_or_path /data/wufan/algos/data/transformers_data/model/layoutxlm-base --output_dir ./output/ --do_train --do_eval --max_steps 1000  --warmup_ratio 0.1 --per_device_train_batch_size 1 --lang zh
```