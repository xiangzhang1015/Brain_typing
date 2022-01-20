# Brain_typing
## Title: Converting your thoughts to texts: Enabling brain typing via deep feature learning of eeg signals

**PDF: [PerCom2018](https://ieeexplore.ieee.org/abstract/document/8444575), [arXiv](https://arxiv.org/abs/1709.08820)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Quan Z Sheng, Salil S Kanhere, Tao Gu, Dalin Zhang**

## Overview
We design a unified deep learning framework that leverages recurrent convolutional neural network to capture spatial dependencies of raw EEG signals based on features extracted by convolutional operations and temporal correlations through RNN architecture, respectively. Moreover, an Autoencoder layer is fused to cope with the possible incomplete and corrupted EEG signals to enhance the robustness of EEG classification.

We also present an operational prototype of a **brain typing system** based on our proposed model, which demonstrates the efficacy and practicality of our approach. [A video demonstrating the system is made available.](https://www.youtube.com/watch?v=Dc0StUPq61k)

## Citing
If you find Brain_typing useful for your research, please consider citing this paper:

    @inproceedings{zhang2018converting,
      title={Converting your thoughts to texts: Enabling brain typing via deep feature learning of eeg signals},
      author={Zhang, Xiang and Yao, Lina and Sheng, Quan Z and Kanhere, Salil S and Gu, Tao and Zhang, Dalin},
      booktitle={2018 IEEE international conference on pervasive computing and communications (PerCom)},
      pages={1--10},
      year={2018},
      organization={IEEE}
    }

## Datasets
Here we provide the datasets used in Brain_typing paper.

[eegmmidb](https://github.com/xiangzhang1015/Brain_typing/blob/master/S1_nolabel6.mat): an example of 1 subject, which is a subset of [Physionet EEG motor movement/imagery database](https://physionet.org/content/eegmmidb/1.0.0/). 

[emotiv](https://github.com/xiangzhang1015/Brain_typing/blob/master/emotiv_7sub_5class.mat): the local real-world
dataset used in this paper. More details about emotive dataset can be found [here](https://github.com/xiangzhang1015/Brain_typing/blob/master/data_readme).

## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.

