# Improving the accuracy of FixMatch method via adding SimCLR method

> This github repository is for Ajou University 2024 summer vacation self-directed research course.
> 
> Advisor: 유종빈 교수님 (jongbinryu@ajou.ac.kr)

## How to run

We only train on CIFAR10 when the number of labeled samples is 40 using WideResNet28-2.
You should check main.py for running baseline code.
This repository's uploaded files are just for FixMatch with add SimCLR method on unlabeled data to few-shot task.

```bash
python main.py +setup=fixmatchsimclr dataset=cifar10 dataset.samples_per_class=4 gpu=6 name=FixMatchSimCLR_UL
```

Also, in this project we use hydra(https://hydra.cc/) for manage configs.

Above bash command, you can easily add configuration with hydra.

Please check in several yaml files in conf/ directory. 

## Experiment Result

|       Model        | Top-1 | Top-5 | # Params | FLOPs | Train time |
|:------------------:|:-----:|:-----:|:--------:|:-----:|:----------:|
| FixMatch(official) | 86.2  |   -   |    -     |   -   |     -      |
| FixMatch(baseline) | 88.1  | 99.6  |  1.47M   | 0.4G  |   17.5h    |


|                   Model                   | Top-1 | Top-5 | # Params | FLOPs | Train time |
|:-----------------------------------------:|:-----:|:-----:|:--------:|:-----:|:----------:|
|  FixMatch with SimCLR at supervised data  |   -   |   -   |    -     |   -   |     -      |
| FixMatch with SimCLR at unsupervised data |   -   |   -   |    -     |   -   |     -      |

## Reference

### Code

#### Base code of bellow files are under the license by Copyright (c) 2022 Seungmin Oh(Ajou University LAB-LVM), MIT License

**If some changes in base code, and not appeared files in below, I mentioned with the comments at codes files.**

> configs
<details>
<summary>fold/unfold</summary>

> - dataset
>> - augmentation
>>> - base_augmentation.yaml
>> - cifar10.yaml, cifar100.yaml, imagenet.yaml
> - info
>> - info.yaml
> - model
>> - base_model.yaml, wrn.yaml
> - setup
>> - fixmatch.yaml, resnet50_cifar.yaml
> - train
>> - optimizer
>>> - adamw.yaml, base_optim.yaml, lamb.yaml, lion.yaml, sgd.yaml
>> - scheduler
>>> - base_scheduler.yaml, cosine.yaml, multistep.yaml, onecyclelr.yaml
>> - base_train.yaml
> - config.yaml

</details> 

> src

<details>
<summary>fold/unfold</summary>

> - data
>> - \_\_init\_\_.py, dataloader.py, dataset.py, randaug.py, transforms.py
> - engine
>> - \_\_init\_\_.py, base_engine.py, fixmatch_engine.py
> - initialize
>> - \_\_init\_\_.py, callback.py, factory.py, initial_setting.py, logger.py
> - misc
>> - \_\_init\_\_.py, metadata.py, pretty_print.py
> - models
>> - \_\_init\_\_.py, for_cifar.py, my_model.py, wide_resnet.py
> - utils
>> - \_\_init\_\_.py, load_checkpoint.py
> - \_\_init\_\_.py


</details>

> main.py, model_spec.py, script.yaml

[1] [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/pdf/2001.07685)

[2] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)

[3] https://github.com/sthalles/SimCLR/tree/master