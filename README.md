# Robust Representation learning by Clustering with Bisimulation Metrics for Visual Reinforcement Learning with Distractions

This is the code of paper 
**Robust Representation learning by Clustering with Bisimulation Metrics for Visual Reinforcement Learning with Distractions**. 
Qiyuan Liu, Qi Zhou, Rui Yang, Jie Wang*. AAAI 2023. 

## Requirements
Since CBM is tested under [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control), you need to download the [DAVIS 2017 dataset](https://davischallenge.org/davis2017/code.html). Make sure to select the 2017 TrainVal - Images and Annotations (480p). The training images will be used as distracting backgrounds. Other dependencies is recorded in the file `requirements.txt`.
```
pip install -r requirements.txt
```

## Reproduce the Results
For example, test the performance of CBM combined with DrQ-v2 on Cartpole Swingup under Distracting Control Suite "easy" distractions, run
```
python scripts/run.py configs/cbm_drqv2_easy/cartpole.json
```

## Citation
If you find this code useful, please consider citing the following paper.
```
@article{liu2023robust,
  title={Robust Representation Learning by Clustering with Bisimulation Metrics for Visual Reinforcement Learning with Distractions},
  author={Liu, Qiyuan and Zhou, Qi and Yang, Rui and Wang, Jie},
  journal={arXiv preprint arXiv:2302.12003},
  year={2023}
}
```

## Acknowledgement
We use the environment wrapper from [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control) and [DrQ-v2](https://github.com/facebookresearch/drqv2). Thanks for their contributions.
