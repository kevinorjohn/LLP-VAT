# LLP-VAT

Pytorch implementation of LLP-VAT

* Kuen-Han Tsai and Hsuan-Tien Lin. Learning from label proportions with consistency regularization. *In Proceedings of the Asian Conference on Machine Learning (ACML)*, November 2020 [ [pdf](https://www.csie.ntu.edu.tw/~htlin/paper/doc/acml20llpvat.pdf) ]

## Environment

* Python version: 3.6.2
* GPU: GeForce GTX 1080
* Prerequisite:
    ```
    pip install -r requirements.txt
    ```


## Usage

Make sure to generate the LLP data before running the experiment of LLP-VAT.

### Preprocessing
```
python -m llp_vat.preprocessing --dataset_name cifar10 --alg uniform --bag_size 64
```

Required arguments:
| Parameter | Description |
|:----------|:------------|
| --dataset_name | `svhn`, `cifar10` or `cifar100` |
| --alg | the bag creation algorithm, `uniform` or `kmeans` |

Optional arguments:
| Parameter | Description |
|:----------|:------------|
| --obj_dir | path to the proccessed object directory|
| --dataset_dir | path to the raw data directory |

Arugments for the bag creation algorithm:
| Algorithm | Parameter | Description |
|:----------|:----------|:------------|
| uniform | -b, --bag_size | number of instances in each bag |
| uniform | --replacement | whether the sample is with replacement |
| kmeans | --k, --n_clusters | number of clusters to be used |
| kmeans | --reduction | number of dimensions to keep |
| uniform, kmeans | --seed | pass an int for reproducible results |


### Experiment
```
python -m llp_vat.main --dataset_name cifar10 --alg uniform -b 64
```

Required arguments:
| Parameter | Description |
|:----------|-------------|
| --dataset_name | `svhn`, `cifar10` or `cifar100` |
| --alg | the bag creation algorithm, `uniform` or `kmeans` |

Optional arguments:
| Parameter | Description | Default |
|:----------|:------------|:--------|
| --obj_dir | path to the proccessed object directory| ./obj |
| --dataset_dir | path to the raw data directory | ./obj/dataset |
| --result_dir | path to the result directory | ./results |
| --num_epochs | number of training epochs | 400 |
| --lr | value of learning rate | 0.0003 |
| --optimizer | `adam` or `sgd` | adam |
| --valid | ratio of the validation set | 0.1 |
| --seed | pass an int for reproducible results | 0 |
| --consistency_type | `vat`, `pi` or `none` | vat |
| --consistency | consistecny loss weight | 0.05 |


## Citation
```
@InProceedings{pmlr-v129-tsai20a, 
    title = {Learning from Label Proportions with Consistency Regularization}, 
    author = {Tsai, Kuen-Han and Lin, Hsuan-Tien}, 
    booktitle = {Proceedings of The 12th Asian Conference on Machine Learning}, 
    year = {2020} 
}
```
