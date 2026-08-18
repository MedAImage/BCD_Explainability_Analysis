# Explainability-aware evaluation of CNNs for breast lesion detection
Official implementation of the paper "Explainability-aware evaluation of CNNs for breast lesion detection"

This work presents a framework to evaluate CNN-based models that jointly considers predictive performance and spatial explainability. In particular, we define an explainability score that quantifies the extent to which relevant activations are concentrated within expert-annotated regions. Based on this score, we reformulate conventional evaluation metrics to incorporate spatial explainability into performance assessment. Since evaluation reliability depends on the fidelity of the explanation maps, we also introduce an attention-based architecture that generates spatial contribution maps directly linked to the prediction process. 


This repository contains all the necessary code to reproduce our experiments on breast lesion detection.
Data and results can be downloaded using this [link](https://unexes-my.sharepoint.com/:f:/g/personal/pilarb_unex_es/IgCqpDamD-eLS72AecX2B8hUARQbNI5aEcKYVoxqR2SXn6M?e=EVqQyc).



## 1. Dataset and Format

The dataset used is composed of screening mammograms from two published datasets: **INbreast** and **VinDrMammo**.
The annotations of the selected images of both dataset are available in a file called *joined_inbreast_vindr.json*.

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

The lesions are Nodulo (mass), Distorsion_arq (architectural distortion), Densidad_asim_foc (focal asymmetry), Microcalcificaciones(Suspicious Calcification), Calc_tip_benig (Suspicious Calcification).
For the existing lesions, the field "label" includes a list of the regions where they are located (x, y, w, h).


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

### Data Augmentation and geometry transformation

This repository contains two types of transformation to enhance the performance of the system:

1. In `data_augmentation` we have the necessary scripts to apply geometrical transformations to the data for oversampling. The `flipImages.py` script allows us to create a flipped version of every mammogram on the dataset, distinguishing them from the originals by appending `_V2` to its name. The other script `expandImages.py` applies an expansion and contraction filter on the breast to generate new examples from original data.
2. In `utils` we include tools to apply specific transformations to highlight the target lesion:
  - `enhance_uniform.py` provides a filter to highlight suspicious structures, focusing on masses. To achieve this, the script applies different computer vision techniques, such as tissue masks, CLAHE, etc.
  - `transforms.py` provides different augmentation techniques and morphological filters specified in the configuration file.





### Configuration-Driven Data Augmentation

We use different configuration `YAML` files to perform channel-wise and lesion-wise data augmentation techniques.

* For each channel in the image, apply different techniques independently, building a 3-channel mammography image where each channel has a different process.
```yaml
channels:
  - {Copy: 100}
  - {Clahe: 100}
  - {EnhanceUniform: 100} # Or TopHat in Microcalcifications
```
* We use oversampling techniques to increase the number of samples for better performance.
```yaml
flipped: True
expanded: True
expanded_cropped : False
expand_factors_cropped: [1.25, 0.8]
testDebug: True
```
In the `configuration_files/augment_transform_` directory there are several `YAML` example files to load and modify.


## Training
### Models structures
This repository contains some of the most commonly used models in medical imaging classification to train with the combined dataset, all of them included in `models/base_models_final.py`, all of them use pre-trained weights of `IMAGENET1K_V1` except CustomResNetBinary50, which uses `IMAGENET1K_V2`.

* CustomResNetBinary: Based on ResNet-18.
* CustomResNetBinary50: Based on ResNet-50.
* CustomDenseNet: Based on DenseNet-121.
* CustomMobileNetV3: Based on MobileNetV3 Large.
* EfficientNetB0: Based on EfficientNet-B0.

Each model has been modified integrating an `Attention-based MIL Head` replacing the global average pooling layer, allowing the model to generate the binary classification and the spatial heatmaps at the same time.

The input to the models is a pytorch tensor of shape [B, 3, H, W] (Batch_size, Channels, Height, Width). The outputs of the model are:

1. `Logit`: The prediction of the model, with shape [B, 1].
2. `map_att`: The attention map from the attentional layer, with shape [B,1,H',W'].
3. `map_imp`: Importance map or Contribution map, calculated by multiplying the attention map by the linear classifier weights and passing the result through a ReLU function, it has the same shape as `map_att`.

### Baseline: Model training
The `train.py` script executes all the training process, this script is designed to be executed on terminal through arguments (`argparse`). This makes modifying the training conditions much easier.

The custom dataset is loaded by the `lesionDataset` class defined in `dataset_load/dataset.py`. The main procedure is loading the json file from the k-fold split.



To execute a single training process you can write on terminal:

```bash
python train.py --trainset <path> --valset <path> --testset <path> --config_file <path> --batch_size <value> --number_of_epochs <value> --learning_rate <value> --model <string value> [ADDITIONAL_ARGUMENTS]
```
The arguments in the example above are the basics to execute a baseline. To ensure that you are training a baseline project you can modify the configuration file that you are using mentioned in `configuration_files/augment_transform_`.

This script generates a `.pth` file with the best weights trained, and are saved in the `bestModels` directory with the following structure:

```bash
bestModels/
└── Chosen_seed/
    └── Positive_class_name/
        ├── EfficientNetB0_seed_17143_copy_clahe_topHat5x5_K1.pth
        ├── CustomMobileNetV3_seed_17143_copy_clahe_topHat5x5_K1.pth
        .
        .
        .
        └── CustomResNetBinary_seed_17143_copy_clahe_topHat5x5_K5.pth
```      




## Evaluation

Once the model is trained, the script `get_model_metrics.py` is executed to load the best model and configs by `argparse` and configuration file in `configuration_files/augment_transform_` to generate the metrics for the subsequent analysis of the results.

To execute a single test process you can write on terminal:

```bash
python get_model_metrics.py --testset <path> --model <string value> --metrics_run_path <path> --json_suffix <string value> --augmentation_config_path <path>
```
These arguments, as well as in `train.py`, are the minimal editable ones to make a baseline execution. All other arguments are detailed in their "help" line inside the code.

This script returns the necessary tools to make the propper evaluation and analysis of the best possible model:
* `Final_metrics_runs.jsonl`: Or `Final_metrics_runs_{suffix}.jsonl` if you use the argument. this file contains all the calculated metrics for the analysis(traditional quantitative metrics and explainability metrics).
* `false_positives.txt`: Additional file generated with the list of the names that were classified as positives when they were negatives, for debugging purpose only.
* Generate images in a folder called `./images_against_posthoc/`, this folder includes the original mammographies and the different heatmaps generated when the visualization is activated and decided to save the files pressing `s` key.


### Automated K-Fold Execution Pipeline: K-Fold model training and inference

This section describes how to execute the full pipeline to conduct the whole experiments using the K-Fold Cross_Validation through training and use them in inference.

The `run_trainings_kfold.py` is the principal script to execute training runs, its function is to perform and execute `train.py` through all K-Folds, models, seeds and configuration files mentioned before and implemented in the code, saving all data needed to perform the inference step. The script needs two key arguments to run: the positive class and the GPU device number

```bash
python run_trainings_kfold.py --positive_class <string value> --device <string value>
```

For the inference step, the `run_testmodel.py` script execute `get_model_metrics.py` in the same way that `run_trainings_kfold.py`, loading all the variables, methods and paths needed from `train.py` and the best models from the `bestModels/` folder.

```bash
python run_testmodel.py --models_path <path> --positive_class <string value>
```

For each run of this `run_testmodel.py` a `.json` file described in `baseline` section is generated and saved in the `Metrics_run` directory with the following structure:



```bash
Metrics_run/
└── Chosen_seed/
    └── Positive_class_name/
        ├── Final_metrics_runs_copy_clahe_topHat5x5_K1.jsonl
        ├── Final_metrics_runs_copy_clahe_topHat5x5_K2.jsonl
        .
        .
        .
        └── Final_metrics_runs_copy_clahe_topHat_K5.jsonl
```      

## Results' analysis

This repository contains several tools to make a visual representation of the metrics in the json files from `Metrics_runs/`.

* `results_analysis.py`: This is the key script for the analysis, works as an internal library and consist of reading the metrics information from the `.json` files and return the python structures with all the data for the rest of scripts to process it. Its output is a nested dictionary divided in 3 different blocks.
  1. standard: For classic quantitative metrics (Precision, Recall, F1-Score, Acc, AUC, AUPRC).
  2. explain: For the quantitative metrics weighted by the energy threshold.
  3. XAI: For the qualitative explainability metrics (Pointing Game, Energy, Correlations, etc.)
* `analyze_performance_results.py`: Evaluates the clinical mean performance of the models and analyse how much this performance fades when explainability metrics weigh the classical metrics. To illustrate this, the script generates a LaTeX table with all numeric results, a bar charts to compare standard performance versus the weighed by the explainability, and heatmaps detailing the penalization of F1-Score and AUC per architecture, threshold and data configurations.
* `analyze_stability_results.py`: Evaluates the stability and robustness of the models analysing through the different K-Folds and seeds. To visualize this, the script generates a heatmap comparing the standard deviation from F1-Score and the AUC of the standard prediction versus the weighed by explainability. Also generates swarmplots to detect the distribution and subsequent dispersion for each point of data, allowing to check if a model is stable and consistent through all possible configuration runs. 
* `analyze_xai_results.py`: Evaluates the precision and quality of the XAI methods, analysing how much the heatmaps correspond to the real medical lesion coordinates. For this purpose, generates a LaTeX table with the mean and standard deviation from Spearman correlation for each architecture and explainability method. Adding to this, a boxplot compares the pointing game accuracy with the post-hoc methods and  a swarmplot is created to visualize which fraction of energy from each map is inside the real lesion region through  different thresholds.


If you need any help or have any suggestions, please contact us:

* Pilar Bachiller-Burgos: pilarb@unex.es
* Jose Luis García-Salas: joselgs96@unex.es




