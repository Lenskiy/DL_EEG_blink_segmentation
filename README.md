# Deep Learning for EEG Blink Detection

This repository implements various deep learning architectures for detecting blinks in EEG signals, comparing performance between healthy subjects and Parkinson's disease patients.

## Overview

Blinks in electroencephalography (EEG) are traditionally treated as unwanted artifacts. However, blink rate and variability serve as important physiological markers for:
- Cognitive load monitoring
- Attention assessment
- Neurological disorder detection

## Datasets

The project supports two datasets:
1. [DS002778](https://openneuro.org/datasets/ds002778/versions/1.0.4)
   - 15 Parkinson's disease patients
   - 16 healthy controls
   - Recorded during resting state
   - Step length: 64 samples

2. [DS004584](https://openneuro.org/datasets/ds004584)
   - 100 Parkinson's disease patients
   - 49 healthy controls
   - Step length: 125 samples

## Model Architectures

The repository implements and compares 10 different architectures:

### Basic Architectures
1. RNN Variants
   - LSTM
   - GRU
   - BiLSTM
   - BiGRU

2. CNN Models
   - Standard CNN
   - Depth-wise CNN

3. TCN Models
   - Standard TCN
   - Depth-wise TCN

4. Hybrid Architectures
   - CNN-RNN (Standard & Depth-wise)
   - TCN-RNN (Standard & Depth-wise)

5. Transformer-based Model

## Hyperparameter Search

Each architecture has specific hyperparameter ranges:

### RNN Models
- Number of units: [8, 16, 32]
- Number of blocks: 1-4
- Unit types: bigruLayer, bilstmLayer, gruLayer, lstmLayer

### CNN/TCN Models
- Filter sizes: [5, 11, 15]
- Number of filters: [8, 16, 32]
- Number of blocks: 1-4

### Transformer Model
- Embedding dimensions: [16, 32, 48]
- Number of heads: [2, 4]
- Number of blocks: 1-3

## Implementation Details

### Data Processing
- Supports 1, 3, or 5 frontal electrodes
- Option for splitting records into smaller overlapping segments
- Configurable step length based on dataset

### Training Configuration
- Optimizer: Adam
- Initial learning rate: 5e-2
- Learning rate schedule: piecewise
- Validation frequency: 50 iterations
- GPU acceleration support
- L2 regularization
- Dropout regularization (12.5%)

## Project Structure
Main scripts 
```plaintext
├── Models/                          # Model architectures
│   ├── constructTCN.m              # TCN model implementation
│   ├── constructRNN.m              # RNN variants implementation
│   ├── constructCNN.m              # CNN implementation
│   └── constructTransformer.m      # Transformer implementation
│
├── Utils/                          # Utility functions
│   ├── gridSearch.m                # Hyperparameter search
│   ├── performanceMetrics.m        # Evaluation metrics
│   └── dataPreprocessing.m         # Data preparation
│
├── Data/                           # Dataset storage
│   ├── ds002778_prepared/
│   │   ├── Train/
│   │   └── Test/
│   └── ds004584_prepared/
│       ├── Train/
│       └── Test/
│
├── Results/                        # Results storage
│   ├── ds002778/
│   └── ds004584/
│
├── hyperParameterSearch_ds002778.mlx   # Main script for model training and evaluation
├── dataPreparationScript_ds002778.mlx  # Data preprocessing and preparation script
└── README.md
```

## Usage

```matlab
numChannels = 5;

options = trainingOptions('adam', ...
    InitialLearnRate    = 5e-2, ....
    LearnRateDropFactor = 0.677, ...
    LearnRateSchedule   = "piecewise", ...
    SequenceLength      = 'shortest',...
    ExecutionEnvironment= 'gpu',...
    L2Regularization    = 0,...
    ValidationPatience  = inf,...
    OutputNetwork       = 'last-iteration',...
    ValidationData      = {XSearchTestPrepared,  YSearchTestPrepared}, ...
    ValidationFrequency = 50,...
    Verbose             = false,...
    Shuffle             = "every-epoch",...
    Plots               = "none");


commonHParameters = struct(...
    'numLabels',        3,... % i.e. {'blink', 'n/a', 'muscle-artifact'}
    'numChannels',      numChannels,... % Channels are stored in the followign order ["Fp1", "Fp2", "Fz", "AF3", "AF4"]
    'dropoutFactor',    0.125,...
    'filterSize',       [5, 11, 15],...
    'numFilters',       2.^(3:5),...
    'numBlocks',        1:4);


tcnResults = gridSearch(@constructTCN, XSearchTrainPrepared, YSearchTrainPrepared, XSearchTrainID, ...
                                       XSearchTestPrepared,  YSearchTestPrepared,  XSearchTestID,...
                                       commonHParameters, options, numTrials, stepLength,...
                                        ["numBlocks", "numFilters", "filterSize"]);
```

## Dependencies

* MATLAB R2023b or later
* Deep Learning Toolbox
* Signal Processing Toolbox
* Parallel Computing Toolbox (optional, for faster training)