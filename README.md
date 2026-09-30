![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)

This master repository contains the official submissions for two advanced machine learning and cognitive science challenges: **I Spy with My Neural Eye** (Diabetic Retinopathy Grading) and **Neuro Decoder** (EEG Cognitive Workload Classification).

---

## 👁 Task 1: I Spy with My Neural Eye (Diabetic Retinopathy Grading)

### 📌 Problem Explanation
Diabetic Retinopathy diagnosis via Retinal Fundus Photography requires models to spot microscopic pixel-level anomalies. Standard images contain vast amounts of wasted space (dark borders) and healthy tissue. The goal is to build an efficient pipeline that focuses strictly on the regions of interest, mimicking human biological vision where uninformative pixels are ignored to maximize computational efficiency and accuracy. 

### 🧠 Solution & Methodology
The solution leverages a deep learning vision pipeline built with PyTorch, utilizing a modified **DenseNet121** architecture[cite: 49]. 

1. **Intelligent Preprocessing (crop_and_enhance):**
   * The pipeline automatically identifies and crops out the useless dark borders using contour detection (`cv2.boundingRect`)[cite: 49].
   * It applies a Ben Graham-style image enhancement by blending the original image with a Gaussian blurred version (`cv2.addWeighted`), which highlights the blood vessels and critical pathologies[cite: 49, 50].
2. **Model Architecture & Training:**
   * Uses a pre-trained `densenet121` with a custom dropout (p=0.4) and a linear layer outputting a single continuous value[cite: 49].
   * The model is optimized using `AdamW` and evaluated using the **Quadratic Weighted Kappa (QWK)** metric, which heavily penalizes predictions that are further from the true ordinal class[cite: 49].
   * Training leverages PyTorch's Automatic Mixed Precision (`autocast`, `GradScaler`) for efficient GPU memory usage[cite: 49].
3. **Inference Pipeline:**
   * A dedicated inference script processes the hidden test folder, generates predictions, clips them to the 0-4 severity scale, and outputs a properly formatted `submission.csv` for Kaggle evaluation[cite: 50].

---

## 🧠 Task 2: Neuro Decoder (EEG Workload Classification)

### 📌 Problem Explanation
This task requires building a binary classifier to distinguish between low cognitive load and high cognitive load brain states using multi-channel EEG recordings from the STEW Dataset. It necessitates raw signal processing, neurological feature extraction, and traditional machine learning to decode complex brain activity.

### 🛠️ Solution & Methodology
The solution is a comprehensive end-to-end signal processing and classification pipeline that maps raw EEG waves to cognitive states[cite: 51].

1. **Preprocessing & Artifact Handling:**
   * **High-pass filter (0.5 Hz):** Applied to remove baseline drift[cite: 51].
   * **Low-pass filter (50 Hz):** Applied to remove high-frequency noise[cite: 51].
   * **Artifact Rejection:** Discarded any epochs where the amplitude exceeded $\pm100\mu V$ to filter out eye blinks and muscle noise[cite: 51].
2. **Neuro-Feature Extraction:**
   * **Frequency Domain:** Extracted band power for Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), and Gamma (30-50 Hz) using Welch's method[cite: 51].
   * **Ratios:** Calculated critical cognitive load markers, most notably the **Theta/Alpha ratio** and (Theta+Beta)/Alpha[cite: 51].
   * **Time Domain & Connectivity:** Computed Hjorth parameters (activity, mobility, complexity), signal entropy, and inter-channel coherence between frontal (Fp1) and parietal (P3) nodes[cite: 51].
3. **Classification:**
   * Utilized an **AdaBoost Classifier** (100 estimators, learning rate = 0.8) trained on the extracted features[cite: 51].
   * Achieved a **5-Fold Cross-Validation accuracy of 75.32%** and a **Test Accuracy of 68.42%** (Precision: 71.43%, Recall: 55.56%, F1-Score: 62.50%)[cite: 51].
4. **Neuroscientific Interpretation:**
   * The **Theta/Alpha ratio** emerged as the strongest predictor of cognitive load[cite: 51].
   * The model correctly mapped biological markers to predictions: Theta band power increased during high cognitive load (indicating working memory engagement), while Alpha band power decreased (indicating task-related attention and reduced cortical relaxation)[cite: 51].
   * These findings heavily align with established neuroscience literature regarding P3b findings and alpha suppression[cite: 51].
