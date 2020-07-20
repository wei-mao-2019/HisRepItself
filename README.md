## History Repeats Itself: Human Motion Prediction via Motion Attention
This is the code for the paper

Wei Mao, Miaomiao Liu, Mathieu Salzmann, Hongdong Li. 
[_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://github.com/wei-mao-2019/HisRepItself). In ECCV 20.

### Dependencies

* cuda 10.0
* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) >1.0.0 (Tested on 1.1.0 and 1.3.0)

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[AMASS](https://amass.is.tue.mpg.de/en) from their official website..

Directory structure:
```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ./datasets directory.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train,
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66
```
```bash
python main_h36m_ang.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 48
```
```bash
python main_amass_3d.py --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 5 --batch_size 128 --test_batch_size 128 --in_features 54 
```
### Evaluation
To evaluate the pretrained model,
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --ckpt ./checkpoint/pretrained/h36m_3d_in50_out10_dctn20/
```
```bash
python main_h36m_ang_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 48 --ckpt ./checkpoint/pretrained/h36m_ang_in50_out10_dctn20/
```
```bash
python main_amass_3d_eval.py --is_eval --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 5 --batch_size 128 --test_batch_size 128 --in_features 54 --ckpt ./checkpoint/pretrained/amass_3d_in50_out25_dctn30/
```

### Citing

If you use our code, please cite our work

```
@inproceedings{wei2020his,
  title={History Repeats Itself: Human Motion Prediction via Motion Attention},
  author={Wei, Mao and Miaomiao, Liu and Mathieu, Salzemann},
  booktitle={ECCV},
  year={2020}
}
```

### Acknowledgments
The overall code framework (dataloading, training, testing etc.) is adapted from [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline). 

The predictor model code is adapted from [LTD](https://github.com/wei-mao-2019/LearnTrajDep).

Some of our evaluation code and data process code was adapted/ported from [Residual Sup. RNN](https://github.com/una-dinosauria/human-motion-prediction) by [Julieta](https://github.com/una-dinosauria). 

### Licence
MIT
