# M-LSD: Towards Light-weight and Real-time Line Segment Detection

Pytorch implementation with training code. <br/>
(this is the sub project for https://github.com/lhwcv/mlsd_pytorch)

## result

model| img_size| sAP10
|---|---|:---:| 
mlsd_tiny (this repo)| 512|  56.4
mlsd_tiny (in the paper)| 512|  58.0
mlsd_large (this repo)| 512|  59.6
mlsd_large (in the paper)| 512|  62.1

(this repo use:  min_score=0.05,  min_len=5, tok_k_lines= 500)

## differences
Due to no official opensource training code, I try to reproduce,<br/>
but not get a good result according to the paper, so I make some differences. <br/>
(Looking forward official opensource training code)  <br/>

main differences compared to the paper: (the up result)

- center map use Focal Loss instead of WCE  (can modify to WCE)
- Use Step LR instead of Cosine .. (can modify to the later)
- Use deconv instead of upsample in tiny model
- No matching loss (I guess my matching loss has BUGS)
- Batch size = 24 ( 64 in the paper, large batch size may good)

## val
(pretrained models put in workdir/models)

eval tiny: 

```
python mlsd_pytorch/pred_and_eval_sAP.py
```
(modify some args can eval large)

## train

### Data Preparation
(You can also follow [AFM](https://github.com/cherubicXN/afm_cvpr2019 ) or others,almost the same )
- Download the [Wireframe dataset](https://github.com/huangkuns/wireframe) and the [YorkUrban dataset](http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) from their project pages.
- Download the JSON-format annotations ([Google Drive](https://drive.google.com/file/d/15z3-xgIzj_-9bep8l6s8dgIbpjKp_8VK/view?usp=sharing)).
- Place the images to "data/wireframe_raw/images/"
- Unzip the json-format annotations to "data/wireframe_raw/"

The structure of the data folder should be
```shell
data/
   wireframe_raw/images/*.png
   wireframe_raw/train.json
   wireframe_raw/valid.json

```
### Train
tiny:
```

python mlsd_pytorch/train.py \
 --config mlsd_pytorch/configs/mobilev2_mlsd_tiny_512_base2_bsize24.yaml
```

large:
```

python mlsd_pytorch/train.py \
 --config mlsd_pytorch/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml
```