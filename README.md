# JEANS
This repo contains code for Cross-lingual Entity Alignment for Knowledge Graphs with Incidental Supervision from Free Text

## Install the dependency 
```
pip install -r requirements.txt
```


## Data and Models
We provide monolingual embeddings used in our experiments and release the pretrained models: https://drive.google.com/drive/folders/1N52NGr5YgL0zydiWM-ESj4vbYIrEpGCO?usp=sharing, https://drive.google.com/file/d/191lP_qgxHrmQ2A6Cupid0_q8kOSAgeFO/view?usp=sharing

Please put the three folders in the data directory

For the original dataset, please refer to: 

https://github.com/nju-websoft/JAPE



## Run the experiments
To run the experiments, use:
```
cd src
./run.sh
```

To use pretrained models, please set restore to True and specify the load_path in the argument
```
cd src
python train_MUSE.py --restore True --load_path "pretrained_model_path"
```
