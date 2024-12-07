# ITGen

This is the codebase for the paper "Iterative Generation of Adversarial Example for Deep Code Models".

## Attack Approach
- [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/)
- [Beam-Attack](https://github.com/CGCL-codes/Attack_PTMC)

## Experiments


**Create Environment**

```
pip install -r requirements.txt
```

**Build tree-sitter**

We use `tree-sitter` to parse code snippets and extract variable names. You need to go to `./python_parser/parser_folder` folder and build tree-sitter using the following commands:

```
bash build.sh
```

**Model Fine-tuning**

Use `train.py` to train models.

Take an example:

```
cd CodeBERT_adv/Clone-detection/code
python train.py
```

**Running Attacks**

You should download the Dataset and Model from this [url](https://drive.google.com/file/d/1IZXbazNhUdgIs-Yr0flj2yjRMeUvjCHh/view?usp=drive_link) and place the file in the appropriate path. 

Take an example:

```
cd CodeBERT_adv/Clone-detection/attack
python run_xxx.py
```

The `run_xxx.py` here can be `run_itgen.py`, `run_alert.py`, `run_beam.py`

Take `run_itgen.py`  as an example:

```
import os

os.system("CUDA_VISIBLE_DEVICES=2 python attack_itgen.py \
        --output_dir=../saved_models \
        --model_type=roberta \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --csv_store_path result/attack_itgen_all.jsonl \
        --base_model=microsoft/codebert-base-mlm \
        --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
        --block_size 512 \
        --eval_batch_size 2 \
        --seed 123456")
```

Run experiments on other tasks of other models as well.

`./CodeBERT_adv/` contains code for the CodeBERT experiment.


<!-- ### Results

results are in `result/` folder, the structure is as follows.

```
CodeBERT_adv/
|-- Clone-detection
   `-- result
       |-- attack_alert_all.jsonl
       |-- attack_beam_all.jsonl
       `-- attack_itgen_all.jsonl
...
```

Take `attack_itgen_all.jsonl` as an example

The csv file contains 9 columns

- `Index: `The number of each sample, 0-3999.
- `Original Code: `The original code before the attack.
- `Adversarial Code: `The adversarial sample code obtained after the successful attack.
- `Program Length: `The length of the code in code token.
- `Identifier Num: `The number of identifiers that the code can extract to.
- `Replaced Identifiers: `Information about identifier replacement in case of a successful attack.
- `Query Times: `Number of query model per attack.
- `Time Cost: `The time consumed per attack, in minutes.
- `Type: ` `0` if it fails, and the method of attack if it succeeds, e.g. `ALERT`, `Beam`, ... .

### Evaluate Result

Run the python script `eval.py` to get the analysis of csv based on csv.

```
cd evaluation
python eval.py
```

In `eval.py` you should change the jsonl path. And the `eval.py` can evaluate the `attack_alert_all.jsonl`, `attack_beam_all.jsonl`, `attack_itgen_all.jsonl`.  -->


## Acknowledgement

We are very grateful that the authors of CodeBERT, GraphCodeBERT, CodeT5, CodeGPT, PLBART, ALERT, BeamAttack make their code publicly available so that we can build this repository on top of their codes. 
