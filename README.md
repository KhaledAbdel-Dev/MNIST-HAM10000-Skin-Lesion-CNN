# 🩺 Skin Lesion Classification with EfficientNet-B0

### **A Deep Learning Pipeline for HAM10000 Dermatoscopic Image Classification**

**Author:** *Khaled Abdelrahman*
**Course:** *EECE.5440 – Computational Data-Driven Modeling I*
**Instructor:** *Dr. Charles Thompson*

This repository contains my complete end-to-end project for classifying dermatoscopic skin-lesion images using **EfficientNet-B0**, trained on the **HAM10000** dataset.
The project includes **data preprocessing, patient-level splitting, model training, evaluation metrics, interpretability (Grad-CAM), and visualizations**.
A fully detailed summary of this work appears in my final presentation deck: [View Presentation](https://www.canva.com/design/DAG4huFHhn8/736PJGOGWT27VcXYsKcShQ/view?utm_content=DAG4huFHhn8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hb010ebeb77)

---

# 📌 Overview

Skin cancer — especially melanoma — requires early detection to improve outcomes.
Dermatoscopic image interpretation is often subjective, making this a strong candidate for computational modeling.

This project builds a **7-class CNN classifier** using a carefully designed pipeline:

* Image preprocessing aligned with EfficientNet’s ImageNet expectations
* Patient-level splitting to prevent data leakage
* Training EfficientNet-B0 from pretrained weights
* Generating robust evaluation metrics
* Creating professional, presentation-quality visualizations

---

# ⚙️ Pipeline Summary

### **1. Dataset Handling**

* Uses the full **HAM10000** dataset (10,015 images)
* Metadata cleaned and filtered to the 7 diagnostic classes
* **Patient-level splitting** (80/20)

  * Ensures images from the same patient never appear in both train and validation
  * Prevents leakage and mimics clinical use
* Final counts:

  * ~3,000 training images
  * ~800 validation images

---

### **2. Preprocessing**

Each image undergoes:

* Convert pixel intensities from **0–255 → [0–1]**
* Resize to **257×257**, then **random crop to 224×224**
* Random horizontal & vertical flips
* **Normalize using ImageNet mean and std**
* No MixUp / CutMix for baseline interpretability

---

### **3. Model Architecture**

**EfficientNet-B0** was selected for its efficiency and strong texture-learning capabilities.

Key features:

* ~5.3M parameters (lightweight but powerful)
* Stacked **MBConv blocks**
* **Depthwise separable convolutions**
* **Squeeze-and-Excitation attention**
* Pretrained ImageNet weights
* Final head: Global Average Pooling → Dense(7)

---

### **4. Training**

* **Optimizer:** AdamW
* **Loss:** Cross-Entropy
* **Batch Size:** 32
* **Epochs:** 20
* **Scheduler:** Cosine Annealing
* Logs training & validation loss
* Automatically saves the best model checkpoint (`best.pt`)

---

### **5. Evaluation**

This project produces:

* Confusion matrix
* Classification report (precision, recall, F1)
* ROC curves (per-class + macro)
* Reliability plots
* Confidence histograms
* Grad-CAM heatmaps
* Summary statistics (accuracy, macro F1, macro AUC)

---

# 📊 Results

### **Overall Performance**

| Metric        | Score     |
| ------------- | --------- |
| **Accuracy**  | **85.3%** |
| **Macro F1**  | **0.74**  |
| **Macro AUC** | **0.96**  |

### Key Insights

* Strong class separability (high AUC)
* Good balanced performance across classes despite dataset imbalance
* Most confusion occurs between *melanoma* and visually similar benign lesions
* Grad-CAM shows the model focuses on the **lesion core**, which is clinically meaningful
* Slight overconfidence → future calibration suggested

---

# 🚧 Limitations & Future Work

### Current Limitations

* Severe dataset imbalance
* Slight overestimation of prediction confidence
* Input resolution limits global shape cues
* Dataset size is small for deep learning

### Planned Improvements

* **Focal loss** or **class weights** to address imbalance
* **Temperature scaling** for probability calibration
* **Higher resolution inputs (320×320)**
* **MixUp / CutMix** to increase effective dataset size
* Testing alternate backbones (DenseNet, ConvNeXt, ViT)

---

# ▶️ Running the Project

### **1. Train**

```bash
python train_ham10000.py --data_dir archive --run_name baseline_effnetb0
```

### **2. Evaluate**

```bash
python eval_best_and_export.py --run_dir outputs/baseline_effnetb0
```

### **3. Visualize Results**

```bash
python analyze_results.py --run_dir outputs/baseline_effnetb0
```

---

# 🧭 Interpretability Tools

### Run Grad-CAM

```bash
python ham_results_tools/run_gradcam.py \
  --run_dir outputs/baseline_effnetb0 \
  --image path/to/image.jpg
```

### Create Lesion Grid

```bash
python ham_results_tools/make_lesion_grid.py \
  --data_dir archive --n_per_class 1 --size 256
```

---

# 🎓 About This Project

This project was completed as a formal presentation for:

**EECE.5440 – Computational Data-Driven Modeling I**
*Fall 2025, UMass Lowell*
Instructor: **Dr. Charles Thompson**

The full presentation deck is included in this repository and demonstrates all results, plots, and model interpretation techniques.

---

# 📬 Contact

If you’re viewing this from my resume or portfolio:

**Email:** [khaled_abdelrahman@student.uml.edu](mailto:khaled_abdelrahman@student.uml.edu)
**LinkedIn:** [https://www.linkedin.com/in/khaled-abdel](https://www.linkedin.com/in/khaled-abdel)
