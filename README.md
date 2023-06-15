# NBTI: NN-Based Typography Incoprating Semantics

**CS470 Introduction to Artificial Intelligence TEAM P12**

![preview_img](https://github.com/DoojinBaek/CS470_Word_As_Image/assets/104518532/4afa0f8c-25b3-4012-83dc-5fea2ca0003c)


## Team Member

| **Name**       | **Student ID** | **github**                     |
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
  <img src="https://github.com/DoojinBaek/CS470_Word_As_Image/assets/104518532/4f686183-f488-4423-bb08-6af2757042b2" width="50%">
</div>


## Run Experiments

```
python code/main.py --experiment <experiment> --semantic_concept <concept> --optimized_letter <letter> --seed <seed> --font <font_name> --use_wandb <0/1> --wandb_user <user name> 
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
* ```--gen_data``` : Default: not generate. generates the data needed for the first learning.


### Examples
1. Applying our encoder and concretizer
```bash
python code/main.py  --semantic_concept "FANCY" --optimized_letter "Y" --font "KaushanScript-Regular" --abstract "TRUE"
```
<br>
<div align="center">
    <img src="https://github.com/DoojinBaek/CS470_Word_As_Image/assets/104518532/c01ca0ed-9f4f-4fa2-b9d5-d83bef37dc4c" width="25%">
</div>
<br>

2. Applying our encoder only
```bash
python code/main.py  --semantic_concept "FANCY" --optimized_letter "Y" --font "KaushanScript-Regular" --abstract "FALSE"
```
<br>
<div align="center">
  <img src="https://github.com/DoojinBaek/CS470_Word_As_Image/assets/104518532/df88eeee-44de-44e3-b665-76a5fa957c8a" width="25%">
</div>

