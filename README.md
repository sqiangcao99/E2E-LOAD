
# E2E-LOAD: End-to-End Long-form Online Action Detection

## Introduction

This is a PyTorch implementation for our ICCV 2023 paper "[`E2E-LOAD: End-to-End Long-form Online Action Detection`](https://arxiv.org/abs/2306.07703)".

![network](assert/network.png?raw=true)

## Environment

This repo is a modification on [PySlowFast](https://github.com/facebookresearch/SlowFast). Please follow their guidelines to prepare the environment.

## Data Preparation

1. Download the [`THUMOS'14`](https://www.crcv.ucf.edu/THUMOS14/) and [`TVSeries`](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html) datasets.

2. Extract video frames at 24 FPS;

3. For constructing the target files, we follow the method used in [LSTR](https://github.com/amazon-science/long-short-term-transformer).

4. The data should be organized according to the following structure. Please modify the root path of the dataset in the configuration file of the corresponding dataset.

    ```
    $DATASET_ROOT
    ├── frames/
    |   ├── video_test_0000004/ (6L images)
    |   |   ├── img_00000.jpg
    |   |   ├── ...
    │   ├── ...
    ├── targets/
    |   ├── video_test_0000004.npy (of size L x 22)
    |   ├──...
    ```

## Training

* Before training, please download the pre-trained model from [MViTv2](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth);

The commands are as follows.

```
REPO_PATH='YOUR_CODE_PATH'
export PYTHONPATH=$PYTHONPATH:$REPO_PATH
python tools/run_net.py --cfg configs/THUMOS/MVITv2_S_16x4.yaml
```

## Online Inference

There are *two kinds* of evaluation methods in our code.

* In order to quickly validate the model's performance during the training process, all test videos are divided into non-overlapping segments, with each segment directly predicting all frames. It is essential to note that these test results are not the final evaluation since most frames do not utilize a sufficiently history during inference. For more detailed information, please refer to [LSTR](https://github.com/amazon-research/long-short-term-transformer#online-inference).

* After the training is completed, you can perform online inference by selecting the best model based on the results obtained during testing while training. This process involves frame-by-frame testing of the entire video, which aligns with real-world applications. Please note that the reported results in our paper are achieved under this mode. You can configure different modes for online testing using the provided configuration file.

    ```
    DEMO:
      ENABLE: True
      INPUT_VIDEO: ['video_validation_0000690'] # Only valid when ALL_TEST=False.
      CACHE_INFERENCE: True # Efficient inference.
      ALL_TEST: True # Test all videos or only one video;
    ```

    The inference commands are as follows:

    ```
    REPO_PATH='YOUR_CODE_PATH'
    export PYTHONPATH=$PYTHONPATH:$REPO_PATH
    python tools/run_net.py --cfg configs/THUMOS/MVITv2_S_16x4_stream.yaml
    ```


## Citations

If you are using the data/code/model provided here in a publication, please cite our paper:
```
@article{cao2023e2e,
    title={E2E-LOAD: End-to-End Long-form Online Action Detection},
    author={Cao, Shuqiang and Luo, Weixin and Wang, Bairui and Zhang, Wei and Ma, Lin},
    journal={arXiv preprint arXiv:2306.07703},
    year={2023}
}
```

## Acknowledge

The project is built upon [MViTv2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) and [LSTR](https://github.com/amazon-science/long-short-term-transformer).
