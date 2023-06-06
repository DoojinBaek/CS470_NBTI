# [NBTI] CS470 Team Project

**CS470 Introduction to Artificial Intelligence TEAM 12**

<img width="1500" alt="NBTI" src=/root/CS470_Final/typography.png>

| **Name**       | **Student ID** |
| :------------- | :------------- |
| Doojin Baek    | 20190289       |
| Min Kim        | 20200072       |
| Dongwoo Moon   | 20200220       |
| Dongjae Lee    | 20200445       |
| Hanbee Jang    | 20200552       |

## Abstract
---

We proposed an NN-based typography model NBTI that can visually represent letters, reflecting the meanings inherent in both concrete and formless words well.



## Run Experiments
---

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
```bash
python code/main.py  --semantic_concept "FANCY" --optimized_letter "Y" --font "KaushanScript-Regular" --abstract "TRUE"
```
