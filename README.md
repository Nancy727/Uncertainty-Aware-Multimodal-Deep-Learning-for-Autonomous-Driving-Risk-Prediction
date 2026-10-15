# Uncertainty-Aware-Multimodal-Deep-Learning-for-Autonomous-Driving-Risk-Prediction

A comprehensive deep learning framework designed to evaluate and predict road risk levels in autonomous driving scenarios by leveraging multimodal data fusion (visual features from **nuImages** combined with sequential ego-vehicle dynamics) alongside **Monte Carlo Dropout uncertainty quantification** for safety-critical reliability.

---

## 📌 Table of Contents
- [About the Project](#-about-the-project)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Model Architectures Explored](#-model-architectures-explored)
- [Experimental Results & Comparison](#-experimental-results--comparison)
- [Uncertainty Quantification (MC Dropout)](#-uncertainty-quantification-mc-dropout)
- [Repository Structure](#-repository-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [Usage Guide](#-usage-guide)
- [Author](#-author)

---

## 🚀 About the Project
In safety-critical autonomous driving systems, reliable risk assessment requires more than just high classification accuracy—it demands calibrated **uncertainty estimation**. This project establishes an end-to-end pipeline using the **nuImages** dataset to categorize driving scenes into three discrete risk levels (**Low**, **Medium**, and **High**) based on vehicle speeds, object counts (pedestrians and vehicles), and bounding box spatial distributions. 

To improve reliability and catch potential false negatives in high-hazard scenarios, the pipeline incorporates **Monte Carlo Dropout (MC Dropout)** to quantify epistemic and predictive uncertainties during inference.

---

## 📊 Dataset & Preprocessing
- **Dataset:** nuImages mini clean split (`nuimages-v1.0-clean`), containing multi-camera sweeps (`CAM_BACK`, `CAM_FRONT`, `CAM_LEFT`, `CAM_RIGHT`, etc.), camera metadata, ego-poses, and object bounding box annotations.
- **Proxy Risk Heuristics:** Scenes are mapped to discrete risk classes (**Low**, **Medium**, **High**) derived from composite scores evaluating:
  - Ego-vehicle speed ($m/s$)
  - Pedestrian and vehicle counts per frame
  - Maximum normalized bounding box height ratios (indicating proximity to vulnerable road users)
- **Data Splitting:** Group-stratified data splits based on `sample_token` to eliminate data leakage between train, validation, and test partitions (Train: 30 images, Validation: 10 images, Test: 10 images).
- **Class Balancing & Augmentations:** Handled severe class imbalance using computed **balanced cross-entropy class weights** and applied robust image augmentations (Random Horizontal Flip, Color Jitter) for training stability.

---

## 🧠 Model Architectures Explored
The repository benchmarks multiple vision and multimodal backbones trained with early stopping:
1. **Custom CNN:** A lightweight 3-block convolutional neural network built from scratch with batch normalization, max-pooling, and dropout.
2. **ResNet50 (Transfer Learning):** ImageNet-pretrained backbone with frozen feature extractor layers and a fine-tuned classification head.
3. **ResNet50 + Numerical Feature Fusion (Multimodal):** Combines visual feature maps extracted from ResNet50 with an auxiliary dense sub-network processing 5 numerical dynamics features (speed, pedestrian counts, vehicle counts, max pedestrian height ratio, max vehicle height ratio).
4. **EfficientNet-B0:** Pretrained lightweight architecture optimized for feature extraction and fine-tuning.
5. **MobileNetV3-Large:** Efficient architecture tested for edge-device feasibility in autonomous vehicles.
6. **DenseNet121:** Dense connection block architecture evaluating gradient flow and feature reuse.
7. **Vision Transformer (ViT-B/16):** Transformer-based global representation model adapted for driving scene classification.

---

## 📈 Experimental Results & Comparison

| Model | Test Accuracy | Balanced Accuracy | Key Observation / Performance Notes |
| :--- | :---: | :---: | :--- |
| **Custom CNN** | 40.0% | 33.3% | Baseline model trained from scratch; prone to minority class omission. |
| **ResNet50 (Transfer Learning)** | 40.0% | 31.7% | Stable convergence on frozen backbone with fine-tuned linear head. |
| **ResNet50 + Numerical Fusion** | **50.0%** | **40.0%** | **Best performing model.** Incorporating dynamic telemetry (speed & object ratios) alongside visual embeddings significantly improved classification performance. |
| **EfficientNet-B0** | 40.0% | 33.3% | Efficient parameter profile with competitive baseline accuracy. |
| **MobileNetV3-Large** | 20.0% | 13.3% | Lower performance under constrained sample size without extensive hyperparameter tuning. |
| **DenseNet121** | **50.0%** | **38.3%** | Strong feature reuse capabilities matching multimodal top accuracy. |

---

## 🛡️ Uncertainty Quantification (MC Dropout)
To evaluate model confidence and safety reliability, **Monte Carlo Dropout** (running $100$ stochastic forward passes with active dropout layers during evaluation) was applied to the best multimodal fusion model:
- **Mean Predictive Variance:** $0.00094$ (indicating overall high probabilistic consistency across forward passes).
- **Uncertainty vs. Correctness Analysis:** 
  - Correct predictions exhibited a lower mean uncertainty ($\approx 0.00086$).
  - Incorrect predictions / errors displayed elevated epistemic uncertainty ($\approx 0.00104$).
- **Error-Detection AUROC:** Evaluated the efficacy of using MC Dropout variance to flag model errors, achieving an AUROC of **0.64**.

---

## 🛠️ Getting Started & Installation

### Prerequisites

* Python 3.8+
* PyTorch 2.1+ with CUDA support (Google Colab T4 GPU recommended)
* Torchvision, Scikit-learn, Pandas, NumPy, Matplotlib, PIL

### Setup Instructions

1. Clone the repository:
```bash
git clone [https://github.com/Nancy727/Uncertainty-Aware-Multimodal-Deep-Learning-for-Autonomous-Driving-Risk-Prediction.git](https://github.com/Nancy727/Uncertainty-Aware-Multimodal-Deep-Learning-for-Autonomous-Driving-Risk-Prediction.git)

```


2. Navigate into the project folder:
```bash
cd Uncertainty-Aware-Multimodal-Deep-Learning-for-Autonomous-Driving-Risk-Prediction

```


3. Open `ProjectDL.ipynb` in Google Colab or your local Jupyter Lab environment.

---

## 💻 Usage

Run the cells sequentially within **`ProjectDL.ipynb`** to:

1. Extract and audit the nuImages dataset archive (`nuimages-v1.0-mini.tgz`).
2. Generate risk heuristic labels and preprocessed tabular splits.
3. Train and validate all 7 deep learning models (from scratch and transfer learning backbones).
4. Execute multimodal feature fusion and run **MC Dropout** uncertainty estimation loops.

---

## 👤 Author

**Nancy Kumari**

* GitHub: [@Nancy727](https://github.com/Nancy727)

```

```
