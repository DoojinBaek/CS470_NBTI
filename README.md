# NBTI: NN-Based Typography Incoprating Semantics

**CS470 Introduction to Artificial Intelligence TEAM P12** \
[NN-Based Typography Incorporating semantics.pdf](NN-Based%20Typography%20Incorporating%20semantics.pdf)


![preview_img](https://github.com/DoojinBaek/CS470_NBTI/assets/104518532/d671f9de-7176-46d5-af50-3d736ff96a10)

## Team Member

| **Name**       | **Student ID** | **Github**                     |
| :------------- | :------------- | :----------------------------- |
| Doojin Baek    | 20190289       | [DoojinBaek][doojin link]      |
| Min Kim        | 20200072       | [minggg012][minggg012 link]    |
| Dongwoo Moon   | 20200220       | [snaoyam][dongwoo link]        |
| Dongjae Lee    | 20200445       | [duncan020313][dongjae link]   |
| Hanbee Jang    | 20200552       | [janghanbee][janghanbee link]  |

[doojin link]: https://github.com/DoojinBaek
[minggg012 link]: https://github.com/minggg012
[dongwoo link]: https://github.com/snaoyam
[dongjae link]: https://github.com/duncan020313
[janghanbee link]: https://github.com/janghanbee

## Reference Paper
[Word-As-Image for Semantic Typography (SIGGRAPH 2023)](https://github.com/Shiriluz/Word-As-Image/tree/main)

## Abstract

>We proposed an NN-based typography model NBTI that can visually represent letters, reflecting the meanings inherent in both concrete and formless words well.
Our focus was on overcoming the limitations of the previous paper, "Word as Image," and presenting future directions. In the previous paper, the excessive deformation of characters made them unreadable, so the degree of geometric deformation was measured to prevent this. However, this approach limited the expressive capabilities of the characters. We shifted our focus to the readability of the characters. Instead of simply comparing geometric values, we employed a visual model that compared encoded vectors to evaluate how well the characters were recognized, using a metric called "Embedding Loss."
Furthermore, the previous model faced challenges in visualizing shapeless words. To address this, we introduced a preprocessing step using LLM fine-tuning to transform these shapeless words into words with concrete forms. We named the module responsible for this transformation the "Concretizer." We used the GPT 3.5 model, specifically the text-davinci-003 variant, and fine-tuned it with 427 datasets. The hyperparameters used for fine-tuning were as follows. The Concretizer module transforms abstract and shapeless words like "Sweet" and "Idea" into words with clear forms like "Candy" and "Lightbulb."

## Model Structure

<div align="center">
  <img src="https://github.com/DoojinBaek/CS470_NBTI/assets/104518532/c4d7993e-ede4-4618-ae3b-592772f1b9cc" width="50%">
</div>


## Dataset

Letter classifier dataset
```
curl http://143.248.235.11:5000/fontsdataset/dataset.zip -o ./data.zip
```

LLM finetuning dataset
```
./finetuning/finetuning.jsonl
```


## Setup

1. Clone the github repo:
```bash
git clone https://github.com/DoojinBaek/CS470_NBTI
cd CS470_NBTI
```
2. Create a new conda environment and install the libraries:
```bash
conda env create -f word_env.yaml
conda activate word
```
3. Install diffusers:
```bash
pip install diffusers==0.8
pip install transformers scipy ftfy accelerate
```
4. Install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```
5. Execute setup bash file:
```bash
bash setup.sh
```

## Run Experiments

```
python code/main.py --experiment <experiment> --semantic_concept <concept> --optimized_letter <letter> --seed <seed> --font <font_name> --abstract <True/False> --gen_data <True/False> --use_wandb <0/1> --wandb_user <user name> 
```


* ```--semantic_concept``` : the semantic concept to insert
* ```--optimized_letter``` : one letter in the word to optimize
* ```--font``` : font name, ttf file should be located in code/data/fonts/

Optional arguments:
* ```--word``` : The text to work on, default: the semantic concept
* ```--config``` : Path to config file, default: code/config/base.yaml
* ```--experiment``` : You can specify any experiment in the config file, default: conformal_0.5_dist_pixel_100_kernel201
* ```--log_dir``` : Default: output folder
* ```--prompt_suffix``` : Default: "minimal flat 2d vector. lineal color. trending on artstation"
* ```--abstract``` : Whether the input semantic concept is abstract(formless) or not, default: False
* ```--gen_data``` : Generates the data needed for the first learning, default: False
* ```--batch_size``` : Default: 1


### Examples
1. Formless word: Applying our encoder and concretizer
```bash
python code/main.py  --semantic_concept "FANCY" --optimized_letter "Y" --font "KaushanScript-Regular" --abstract "TRUE"
```
<br>
<div align="center">
    <img src="https://github.com/DoojinBaek/CS470_NBTI/assets/104518532/da400674-f354-4e12-b2cd-8f2d77070bd7" width="25%">
</div>

<br>

2. Concrete word: Applying our encoder only
```bash
python code/main.py  --semantic_concept "CAT" --optimized_letter "C" --font "Moonies" --abstract "FALSE"
```
<br>
<div align="center">
  <img src="https://github.com/DoojinBaek/CS470_NBTI/assets/104518532/6cd8a23f-c251-450d-94c7-f4933fd6d552" width="25%">
</div>


