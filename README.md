# Vision Transformers

This repository contains the code and experimental notebooks developed for my diploma thesis on the **evaluation of Vision Transformers and CNN-based architectures for visual understanding**.

The work focuses on a comparative study of modern deep learning architectures for **image classification, object detection, robustness evaluation, and medical image analysis**, with particular emphasis on Vision Transformer (ViT) and Transformer-based models.

> **Thesis project:** Comparative evaluation of CNN, Transformer, and hybrid architectures for image classification, and object detection.

---

## Overview

The repository brings together the experiments, dataset preparation, evaluation scripts, and analysis used throughout the thesis.

The main goal was to investigate how Transformer-based computer vision models compare with conventional CNN architectures across different computer vision tasks and evaluation settings.

The experiments cover:

- Image classification
- Object detection
- Vision Transformers
- CNN architectures
- Transformer-based detectors
- ImageNet-1K evaluation
- COCO object detection
- Robustness to image corruptions
- Medical image classification
- Data augmentation
- MixUp and CutMix
- Quantitative and comparative analysis

---

## Repository Structure

| Directory | Description |
|---|---|
| `Classification/` | Image classification experiments using CNN and Transformer-based architectures |
| `Object Detection/` | Object detection experiments using CNN- and Transformer-based detectors |
| `Analysis/` | Evaluation, comparison, and visualization of experimental results |
| `Robustness/` | Robustness experiments using ImageNet-A, ImageNet-C, and mean Corruption Error (mCE) |
| `Medical/` | Experiments on medical image classification and ROC-AUC evaluation |
| `Medical Data Augmentation/` | Experiments investigating data augmentation, MixUp, and CutMix for medical images |
| `Datasets/` | Dataset preparation, loading, preprocessing, and supporting files |
| `cifar-10.ipynb` | CIFAR-10 experimentation and model evaluation |

---

## Classification

The `Classification/` directory contains experiments comparing CNN-based and Transformer-based approaches for image classification.

### Experiments

- **CNN models**
  - `Cnn_models_classification.ipynb`
- **ImageNet-1K**
  - `Imagenet-1k.ipynb`

The classification experiments investigate model performance and provide the basis for comparing convolutional architectures with Vision Transformer approaches.

---

## Object Detection

The `Object Detection/` directory contains experiments with both conventional CNN-based detectors and Transformer-based object detection architectures.

### Models / Experiments

- CNN-based object detection
- Transformer-based object detection
- **DETR / Deformable DETR**
- **YOLOS**
- Transformer object detection experiments

Relevant notebooks include:

- `CNN_Object_Detection Test.ipynb`
- `Transformer_Object_detection.ipynb`
- `Deformable DETR.ipynb`
- `YoloS.ipynb`

The experiments are primarily evaluated using the **COCO** object detection benchmark and focus on metrics such as mean Average Precision (mAP) and mean Average Recall (mAR).

---

## Robustness Evaluation

The `Robustness/` directory investigates how different vision architectures behave under distribution shifts and image corruptions.

Experiments include:

- **ImageNet-A**
- **ImageNet-C**
- **Mean Corruption Error (mCE)**

These experiments are used to examine model behavior beyond standard clean-data accuracy and to evaluate the effect of common image corruptions and challenging samples on model performance.

---

## Medical Image Analysis

The repository also includes experiments involving medical image classification.

The `Medical/` directory contains:

- ROC-AUC evaluation
- Medical image classification experiments
- Comparative analysis of model performance

The corresponding dataset preparation and preprocessing notebooks can be found under `Datasets/`.

---

## Data Augmentation

The `Medical Data Augmentation/` directory investigates the effect of data augmentation techniques on medical image classification.

Experiments include:

- Standard data augmentation
- **MixUp**
- **CutMix**
- Combinations of augmentation techniques

The main notebooks include:

- `Data Augmentation.ipynb`
- `Medical_roc_auc_Data_Aug.ipynb`
- `Medical_roc_auc_MixUp.ipynb`
- `Medical_roc_auc_CutMix.ipynb`
- `Medical_roc_auc_MixUp+Data_Aug.ipynb`

The experiments evaluate whether augmentation strategies can improve model generalization when working with comparatively limited medical imaging data.

---

## Datasets

The `Datasets/` directory contains notebooks and supporting files used to prepare and work with the datasets required by the experiments.

The repository includes support for:

- **ImageNet-1K / ILSVRC2012**
- **COCO**
- Image corruption datasets
- Medical image datasets
- Dataset preparation for classification
- Dataset preparation for object detection

Dataset-related notebooks include:

- `Imagenet-1k_Dataset.ipynb`
- `Coco_dataset.ipynb`
- `Corruption_dataset.ipynb`
- `Medical_Dataset_classification.ipynb`
- `Medical_Dataset_Object_detection.ipynb`

---

## Experimental Analysis

The `Analysis/` directory contains notebooks used to process and analyse the results obtained from the experiments.

The analysis includes:

- ImageNet-1K results
- ImageNet-A robustness results
- ImageNet-C corruption results
- Object detection results
- Medical image results
- Comparative performance analysis
- Result visualization

---

## Models and Architectures

The experiments cover multiple families of computer vision architectures, including:

### Convolutional Neural Networks

CNN-based architectures are used as conventional baselines for comparison with Transformer-based approaches.

### Vision Transformers

Vision Transformer architectures are evaluated for image classification and visual recognition tasks, with emphasis on the use of self-attention to model relationships between image patches.

### Transformer-based Object Detectors

The repository also evaluates Transformer-based approaches for object detection, including:

- DETR
- Deformable DETR
- YOLOS

This allows comparison between traditional CNN-based detection pipelines and more recent Transformer-based approaches.

---

## Evaluation

Depending on the task, the experiments use metrics such as:

### Classification

- Accuracy
- ROC-AUC
- Error analysis

### Object Detection

- mAP (mean Average Precision)
- mAR (mean Average Recall)
- AP for different object sizes

### Robustness

- Accuracy under image corruptions
- Corruption error
- Mean Corruption Error (mCE)

The analysis notebooks are used to compare these metrics across architectures, datasets, and experimental conditions.

---

## Technologies

The project is primarily implemented in **Python** using the deep learning and scientific computing ecosystem.

Main technologies and libraries include:

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Additional libraries may be used by individual experiments.

---

## Running the Experiments

Clone the repository:

~~~bash
git clone https://github.com/Tsomaros/Vision-Transformers.git
cd Vision-Transformers
~~~

Install the required Python dependencies for the experiment you want to reproduce.

For example:

~~~bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
~~~

Then launch Jupyter:

~~~bash
jupyter notebook
~~~

Open the relevant notebook and follow the cells in order.

> **Note:** Some experiments require external datasets such as ImageNet-1K and COCO. These datasets are not fully distributed with this repository and may require separate download and configuration.

---

## Thesis Scope

The repository represents the practical implementation and experimental part of my diploma thesis.

The overall research workflow can be summarized as:

~~~text
Dataset Preparation
        ↓
Model Selection
        ↓
Training / Inference
        ↓
Performance Evaluation
        ↓
Robustness Testing
        ↓
Comparative Analysis
~~~

The experiments were designed to provide a consistent basis for studying the differences between **CNNs, Vision Transformers, and Transformer-based hybrid architectures** across multiple visual recognition tasks.

---

## Related Work

This repository is associated with my research work on the evaluation of Vision Transformers for image recognition and object detection.

The experiments are intended to accompany the corresponding thesis and research publication.

---

## Author

**Dimitris Vlachogiannis**

GitHub: [@Tsomaros](https://github.com/Tsomaros)

---

## Repository

[https://github.com/Tsomaros/Vision-Transformers](https://github.com/Tsomaros/Vision-Transformers)

---

## License

This repository contains research and educational code developed as part of a diploma thesis. Please check the individual files and referenced projects for their respective licenses and attribution requirements.
