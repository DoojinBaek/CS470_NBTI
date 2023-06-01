# CS470 Team Project Word as Image

python code/main.py --experiment <experiment> --semantic_concept <concept> --optimized_letter <letter> --seed <seed> --font <font_name> --use_wandb <0/1> --wandb_user <user name> 
```
* ```--semantic_concept``` : the semantic concept to insert
* ```--optimized_letter``` : one letter in the word to optimize
* ```--font``` : font name, the <font name>.ttf file should be located in code/data/fonts/

Optional arguments:
* ```--word``` : The text to work on, default: the semantic concept
* ```--config``` : Path to config file, default: code/config/base.yaml
* ```--experiment``` : You can specify any experiment in the config file, default: conformal_0.5_dist_pixel_100_kernel201
* ```--log_dir``` : Default: output folder
* ```--prompt_suffix``` : Default: "minimal flat 2d vector. lineal color. trending on artstation"
* ```--gen_data``` : Default: not generate. 맨 처음 학습에 필요한 데이터를 생성해준다.

### Examples
```bash
python code/main.py  --semantic_concept "BUNNY" --optimized_letter "Y" --font "KaushanScript-Regular" --seed 0