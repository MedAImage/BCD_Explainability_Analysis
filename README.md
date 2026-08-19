# Explainability-aware evaluation of CNNs for breast lesion detection
Official implementation of the paper "Explainability-aware evaluation of CNNs for breast lesion detection"

This work presents a framework to evaluate CNN-based models that jointly considers predictive performance and spatial explainability. In particular, we define an explainability score that quantifies the extent to which relevant activations are concentrated within expert-annotated regions. Based on this score, we reformulate conventional evaluation metrics to incorporate spatial explainability into performance assessment. Since evaluation reliability depends on the fidelity of the explanation maps, we also introduce an attention-based architecture that generates spatial contribution maps directly linked to the prediction process. 


This repository contains all the necessary code to reproduce our experiments for breast lesion detection.
Data and results can be downloaded using this [link](https://unexes-my.sharepoint.com/:f:/g/personal/pilarb_unex_es/IgCqpDamD-eLS72AecX2B8hUARQbNI5aEcKYVoxqR2SXn6M?e=EVqQyc).



## Dataset

The dataset used is composed of screening mammograms from two published datasets: **INbreast** and **VinDrMammo**.
The annotations of the selected images of both datasets are available in a file called *joined_inbreast_vindr.json*.

For each sample, its information is stored as follows:
```json

{
    "5eae9beae14d26fd_L_ML.dcm": {
        "image": "outDat/5eae9beae14d26fd_L_ML.png",
        "label": {
            "Nodulo": [
                [
                    19.148446490218646,
                    1596.980437284235,
                    405.9470655926352,
                    436.5845799769851
                ]
            ],
            "Distorsion_arq": [],
            "Densidad_asim_foc": [],
            "Microcalcificaciones": [],
            "Calc_tip_benig": []
        }
    }


```

The fields composing each entry are:
* The name of the image in dicom format to identify each entry.
* The path to the image in png format (field "image").
* Lists of the different lesions found in the image (field "label").

The lesions are Nodulo (mass), Distorsion_arq (architectural distortion), Densidad_asim_foc (focal asymmetry), Microcalcificaciones(microcalcifications), Calc_tip_benig (Suspicious Calcification).
For the existing lesions, the field "label" includes a list of the image regions where they are located (x, y, w, h).


### Dataset preparation

To prepare the dataset, it is first necessary to download the original data from Vindr-Mammo and INbreast:

* For VinDr-Mammo, you can access the data from here: [https://physionet.org/content/vindr-mammo/1.0.0/](https://physionet.org/content/vindr-mammo/1.0.0/)
* INBreast dataset is available at [https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset)

Once both datasets have been downloaded and unzipped, the images composing our dataset can be extracted using the script 'data_organization/get_dataset_images.py':


```python
cd data_organization
python get_dataset_images.py --dataset DATASET.json --vindrdir VINDR_PATIENTS_DIRECTORY --vindrcsv VINDR_FINDING_ANNOTATIONS_FILE --inbreastdir INBREAST_IMAGES_DIRECTORY --outputdir OUTPUT_DIRECTORY
```

This script creates a folder named "outDat" in the directory specified by the *`outputdir`* argument, containing all the dataset images in PNG format. 

Some of these images contain letters indicating the laterality and projection of the mammogram, which can negatively affect model training. To remove these letters and obtain clean images, use the "cleanLetters.py" script located in "data_organization" as follows:

```python
python cleanLetters.py PATH_TO_THE_DATASET_IMAGES
```

Additionally, our dataset includes several versions of each image as result of applying different geometric transformations. To complete the dataset preparation stage, these additional versions should be generated running the script "data_augmentation/augment_dataset.py":


```python
cd data_augmentation
python augment_dataset.py --dataset DATASET.json --dataroot PARENT_DIRECTORY_OF_outDat
```

### Data splitting
Once the dataset is prepared, it has to be splitted into train/validation/test sets. We provide two different scripts for this splitting in "data_organization": one producing a single partitioning and another one for K-Fold cross validation. To ensure results' reproducibility, both scripts require a random seed as argument. They can be run as follows:

For a single partitioning, run the following command:
```python
cd data_organization
python train_val_test_split.py --dataset DATASET.json --seed SEED --positive_classes LESION_NAME --split_root SPLIT_DIRECTORY
```


Alternatively, a K-Fold split can be generated with "kfold_split.py":

```python
cd data_organization
python kfold_split.py --dataset DATASET.json --seed SEED --positive_classes LESION_NAME --split_root SPLIT_DIRECTORY
```

This command generates a 5-fold stratified split with the following organization:

```bash
json_splits/
└── lesion_name/
    └── chosen_seed/
        ├── K1/
        │   ├── joined_dataset_training_...json
        │   ├── joined_dataset_validation_...json
        │   └── joined_dataset_test_...json
        ├── K2/ 
        ├── K3/ 
        ├── K4/ 
        └── K5/ 
```

The 5-fold split used in our experiments can be downloaded from the link provided in the first section.


## Training
### Models' architecture

This repository implements an attention-based architecture leveraging some of the most commonly used models for classification:

* CustomResNetBinary: Based on ResNet-18.
* CustomResNetBinary50: Based on ResNet-50.
* CustomDenseNet: Based on DenseNet-121.
* CustomMobileNetV3: Based on MobileNetV3 Large.
* EfficientNetB0: Based on EfficientNet-B0.

Each model has been modified by replacing their classification block with an attention-based head that provides a prediction score along with a contribution map that represents the contribution of each region to the model's prediction.

The input of the models is a pytorch tensor of shape [B, 3, H, W] (Batch_size, Channels, Height, Width). The outputs of the model are:

1. `Logit`: The prediction of the model, with shape [B, 1].
2. `map_att`: The attention map from the attentional layer, with shape [B,1,H',W'].
3. `map_cont`: the contribution map, with shape [B,1,H',W'].

### Data configuration
We use different data configurations to analyze how different visual representations of the data affect model training and evaluation. Specifically, at the image level, the channels of the input images can be modified to contain different processed versions of the original mammogram. Likewise, at the dataset level, models can be trained with or without the augmented data provided by the flipped and expanded versions of the images.

The data configuration used for training an evaluation must be specified in a YAML file. The fields in this file are the following:

* channels: list with the processing technique applied to each channel of the image. The different techniques are:
	* Copy: original channel.
	* Clahe: result of applying the CLAHE filter.
	* TopHat5x5: result of applying a white top-hat filter (5x5). It is used to enhance microcalcifications.
	* EnhanceUniform: result of applying a custom filter designed to enhance mass-type lesions.
* flipped: boolean flag to indicate whether or not to use the flipped version of the images of the dataset.
* expanded: boolean flag to indicate whether or not to use the expanded/contracted versions of the images of the dataset.

The configuration_files/augment_transform_ directory includes several data configuration files used in our experiments.


### Model training

To train a model, `use *train.py*, specifying the data and training configuration parameters as argument:

* --trainset: path to the JSON file containing the training set.
* --valset: path to the JSON file containing the validation set. This set is used for early-stopping.
* --dataroot: path to the directory containing the images' folder (*outDat*).
* --model: backbone architecture used (CustomResNetBinary, CustomResNetBinary50, CustomDenseNet, ...)
* --augmentation_config_path: path to the data configuration file.
* --batch_size: batch size used during training.
* --number_of_epochs: maximum number of epochs.
* --patience: number of epochs to wait for loss improvement on the validation set before stopping the training.
* --learning_rate: learning rate.
* --positive_classes: name of the lesion acting as the target class (use *Nodulo* for masses and *microcalcificaciones* for microcalcifications).
* --seed: random seed used to reproduce the stochastic conditions of the training process.
* --seed_split: seed used to generate the data split (used for the organization of the model files).
* --model_save_path: path where the model will be saved (default is *bestModels*).
* --suffix: suffix added to the name of the file.
* --device: device used during the training process (default is *cuda*).

The trained model is saved using the path specified in `*model_save_path*`. To facilitate model organization on disk, this path includes a subdirectory containing the split seed and, within that, another subdirectory named after the lesion. This is the final directory where the model is saved. The name assigned to the file has the following format: {MODEL}\_seed\_{SEED}\_{SUFFIX}.pth, with {MODEL}, {SEED} and {SUFFIX} being the values of the arguments *model*, *seed*, and *suffix*, respectively.

Next, an example of models' organization is shown:
```bash
bestModels/
└── 76014/
    └── Nodulo/
        ├── EfficientNetB0_seed_17143_copy_clahe_topHat5x5_K1.pth
        ├── CustomMobileNetV3_seed_17143_copy_clahe_topHat5x5_K1.pth
        .
        .
        .
        └── CustomResNetBinary_seed_17143_copy_clahe_topHat5x5_K5.pth
```      
### Training models' with K-Fold cross validation

The experiments reported in our work evaluate different model architectures as well as different data and training conditions using K-Fold cross-validation. The trained models are available at this [link](https://unexes-my.sharepoint.com/:f:/g/personal/pilarb_unex_es/IgCqpDamD-eLS72AecX2B8hUARQbNI5aEcKYVoxqR2SXn6M?e=EVqQyc).

To reproduce the experiments, you can run the script *run_trainings_kfold.py* for each type of lesion. The following arguments has to be specified:

* --positive_classes: name of the lesion acting as the positive class.
* --dataroot: path to the directory containing the images' folder (*outDat*).
* --data_split_path: path to the directory containing the K-Fold split.
* --device: device used for training.

The script trains a total of 625 models per lesion type, varying the backbone, training seed, data configuration, and dataset split. Each model's filename includes a suffix indicating the data configuration and the split number. The script can be modified to reduce the number of training runs and thus avoid an excessive number of executions.

## Evaluation

Once a model is trained, it can be evaluated using *get_model_metrics.py*. This script generates a report in JSON format contaning different metrics: standard evaluation metrics (precision, recall, F1-score, ...), explainability-aware versions of those metrics as well as metrics used to assess the fidelity and quality of the generated contribution maps.

To run the script, you must specify the following arguments:

* --testset: path to the JSON file containing the test set.
* --dataroot: path to the directory containing the images' folder (*outDat*).
* --model: backbone architecture.
* --model_weights_path: path to the model file.
* --seed: random seed used during the model training.
* --positive_classes: lesion name used as target class during the model training.
* --augmentation_config_path: data configuration file used for training the model.
* --metrics_run_path: path to the directory where the metrics' file will be saved.
* --json_suffix: optional suffix to be added to the name of the metrics' file.

The metrics are saved in the specified folder with the filename *Final\_metrics\_runs\_{JSON_SUFFIX}.jsonl*, where {JSON_SUFFIX} is the string specified as the *--json_suffix* argument. If the file already exists, the new metrics are appended to it.

Additionally, the script allows for visualizing the generated contribution maps and comparing them with various CAM methods. To do this, you must clone the `[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam.git)` repository into the project's main folder. Furthermore, to obtain fidelity metrics for these additional methods, you need to disable map normalization by commenting out lines 164-171 of the file *pytorch-grad-cam/pytorch_grad_cam/utils/image.py*(function *scale_cam_image*); this normalization is otherwise performed by the script provided in our repository after obtaining some correlation metrics. 

To visualize the maps, it is necessary to add the *--show_maps*`option to the list of arguments. Additionally, to obtain comparative metrics with CAM methods, you must specify the *--compare_cam*` argument. If both arguments are enabled, the CAM maps are included in the visualization.


### Evaluation of all trained models

As with training, all models trained in our experiments for each type of lesion can be evaluated using a single script. Specifically, you can use *run_testmodel.py* with the following arguments: 

* --models_path: path to the directory containing all the trained models.
* --positive_classes: name of the lesion acting as the positive class.
* --dataroot: path to the directory containing the images' folder (*outDat*).
* --data_split_path: path to the directory containing the K-Fold split.
* --device: device used for training.

The script creates a directory called *ALL_metrics* containing a JSONL file for each data split. Each file includes the evaluation metrics for all the models trained using the corresponding split. 

## Results' analysis

This repository contains several tools to make a visual representation of the metrics in the json files from `Metrics_runs/`.

* `analyze_performance_results.py`: analyzes the performance of the models using standard and explainability-aware metrics.To illustrate this, the script generates a table with all numeric results, a bar charts to compare standard and explainability-aware metrics, and heatmaps detailing the penalization of F1-Score and AUC per architecture, energy threshold and data configurations.
* `analyze_stability_results.py`: analyzes the stability and robustness of the models through the different K-Folds and seeds. To visualize this, the script generates a heatmap comparing the standard deviation of F1-Score and AUC in their standard and explainability-aware versions. Also generates swarmplots depicting the effect of different random seeds and data partitioning in the performance of the models.
* `analyze_xai_results.py`: analyzes the fidelity and quality of XAI methods by evaluating the correspondence between explanation maps and model predictions, as well as the alignment of highlighted image regions with the data annotations. For this purpose, it generates a table with the mean and standard deviation of the Spearman correlation for each architecture and explainability method. In addition, it generates a boxplot comparing the pointing game accuracy of the different methods, and a swarmplot showing the ROI energy fraction provided by each explanation map using different architectures and energy thresholds.


If you need any help or have any suggestions, please contact us:

* Pilar Bachiller-Burgos: pilarb@unex.es
* Jose Luis García-Salas: joselgs96@unex.es




