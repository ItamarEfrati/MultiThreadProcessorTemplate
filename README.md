# PipeLion
> ML project template

## Table of Contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Implementation Steps](#implementation-steps)
* [Installation](#installation)
* [Usage](#usage)
* [Example](#example)
<!-- * [License](#license) -->


## General Info
PipeLion Template is designed to provide a ready-to-use structure that integrates both data processing pipelines and hyperparameter studies. This template serves as a foundation for developing scalable, efficient, and customizable workflows for various data-driven tasks, including ETL (Extract, Transform, Load) operations and machine learning model optimization.

The template contains two main components: the data processing pipeline and the hyperparameters search study. The data preprocessing pipeline is designed to manage and execute data processing pipelines using a microservice architecture. Each microservice handles a specific task (e.g., data extraction, transformation, loading, and analytics), allowing for scalability and easy maintenance. The system leverages multiprocess capabilities to ensure efficient use of resources and to handle high-throughput data streams.

A Study allows users to specify a search space for hyperparameters and manages the optimization process, leveraging distributed computing to efficiently explore the search space.

### Key Features
- **Modular Design**: Each stage of the pipeline is a separate microservice, making it easy to add, remove, or update components without affecting the entire system.
- **Scalability**: Microservices can be scaled independently based on the workload, ensuring optimal resource utilization.
- **Fault Tolerance**: The system is designed to handle failures gracefully, with mechanisms for retrying failed tasks and logging errors for later analysis.
- **Asynchronous Processing**: Utilizes message queues to decouple services and enable asynchronous processing, improving overall system performance.
- **Configuration Management**: Pipelines are defined using configuration files, allowing for easy customization and deployment of different workflows.
- **Study Definition**: Users define a Study that includes the hyperparameters to be optimized, their search space, and the objective function to evaluate.
- **Search Algorithms**: Supports various search algorithms such as Random Search, Grid Search, and Bayesian Optimization (Optuna).

### Technologies
- [**Optuna**](https://optuna.readthedocs.io/en/stable/index.html): For hyperparameter optimization.
- [**Hydra**](https://hydra.cc/docs/intro/): For configuration management.
- [**scikit-learn**](https://scikit-learn.org/stable/index.html): For machine learning models and utilities.

## Implementation Steps
1. **Develop Microservices**: Create microservices for each stage of the pipeline.
2. **Create a DataLoader**: Load the features for training.
3. **Create a New Study**: Or use an existing one by choosing the appropriate model and parameters search space.
4. **Create Configuration Files**: Define your pipeline, study (including choosing the Optuna optimization algorithm), and dataloader.
5. **Run `main.py`**: The script will initiate the pipeline project and the study.
6. **Analyze Results**: All pipeline outputs are saved in the `data/preprocess` folder, and all run results are saved in `assets/results`.


## Installation
First, clone the repository to your local machine:
```
git clone https://github.com/itamar.efr/pipelion-template.git
cd pipelion
```

```
pip install -r requirements.txt
```
## Usage
After implementing the required componetns, update the main configuraion file 

```yaml
defaults:
  - _self_
  - hydra: dev
  - pipelines: the name of the pipeline config
  - study: the name of the study config
  - dataloader: the name of the dataloader

preprocess: true to run the preprocess otherwise false to skip
train: true to run the hyperparameters-search otherwise false to skip
seed: 42
```

To run the process just execute main.py

`python src/main.py`

## Example

The code itself comes with an example. The pipeline has one micro-service that measure the size of a given file. As for the study there is an implementations of a
simple dataloader that generate random labels for the study. Below are some examples for configurations for the folder.

### Dataloader
```
_target_: src.training.data_loaders.text_size_data_loader.TextSizeDataLoader
pipeline_type: training
seed: 42
labels_path: # add path for labels for real data
split_data: true
pattern: text_counter_length.txt
```

### Pipeline

```
# In this document we can see the different configurations of the training pipeline.
# The microservices are order by their operating order.

pipeline_executor:
  _target_: src.pipline_executor.PipelineExecutor
  _partial_: True
  input_dir: text
  pipeline_type: training
  is_input_data_in_folder: True


text_parsers:
 text_parser_n_threads: 1
 text_parser:
     _target_: src.micro_services.text_counter.TextCounter
     arg1: 1
     arg2: 2
```

### Study

```
_target_: src.training.hyperparameters_tuning.classification.random_forest.RandomForestStudy
sampler:
  _target_: optuna.samplers.TPESampler
  n_startup_trials: 1
  seed: 42
feature_type: text
seed: 42
direction: maximize
optimize_metric: f1_macro
n_trials: 2
n_jobs: 1
n_jobs_forest: -1
cv:
  _target_: src.training.train_val_splitters.text_size_train_val_splitter.TextSizeTrainValSplitter
```
Note that for cv you can use either the train_val_splitters to split train to train and val (you need to implement), or
you can put any cv object acceptable by scikit-learn.
