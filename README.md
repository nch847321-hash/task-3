# task-3


---

# 🐶🐱 Cats vs Dogs Classification using Support Vector Machine (SVM)

## 📌 Overview

This project implements a **Support Vector Machine (SVM)** to classify images of **cats** and **dogs** from the Kaggle dataset.
The goal is to train a binary classifier that can accurately distinguish between the two classes.

---

## 📂 Dataset

The dataset is the popular **Dogs vs Cats** dataset from Kaggle:

* **Training Data:** Contains labeled images of cats and dogs.
* **Test Data:** Unlabeled images for evaluation.

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib tqdm
```

---

## 🚀 Usage

1. **Download** the dataset from [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and extract it.
2. Place the images in the following structure:

```
dataset/
│── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   ├── dog.1.jpg
│── test/
    ├── 1.jpg
    ├── 2.jpg
```

3. Run the training script:

```bash
python svm_cats_dogs.py
```

---

## 📊 Model Details

The project uses **SVM from scikit-learn**:

```python
from sklearn.svm import SVC
```

**Steps:**

1. Load images and preprocess them (resize, grayscale, flatten).
2. Convert images into numerical feature vectors.
3. Split into training and testing sets.
4. Train an SVM classifier.
5. Evaluate performance on test data.

---

## 📈 Example Output

* Accuracy score on the test set.
* Confusion matrix.
* Sample predictions with images.

Example:

```
Training Accuracy: 98.5%
Test Accuracy: 96.2%
Predicted: Dog | Actual: Dog
Predicted: Cat | Actual: Cat
```

---

## 📌 Future Improvements

* Use **HOG (Histogram of Oriented Gradients)** features for better performance.
* Integrate **deep learning** (CNNs) for higher accuracy.
* Deploy as a **web app** for real-time classification.

---

## 🖋 Author

**Nirmal Chaturvedi**

---

If you want, I can now **write the complete Python code** for loading the Kaggle dataset, training the SVM, and showing predictions.
Do you want me to prepare that next?
