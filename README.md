# AI-Based Indian Sign Language and Accuracy Assessment System

A starter implementation for a Review 1 prototype. It trains a CNN to classify labelled ISL gesture images, evaluates it on a held-out test set, and provides a Streamlit image-upload demo.

## Important limitations
- This baseline recognises image classes only. It does not yet assess movement across frames or non-manual features.
- The dataset is not included. Use a dataset you are permitted to use, and check its licence and label meanings.
- Do not describe scores as "signing accuracy" or as an assessment of all five parameters. They are image-classification results until further components are implemented and validated.

## 1. Create an environment
Python 3.10 or 3.11 is recommended.

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Put the dataset in this structure
Each class must have its own folder. The folder name becomes the class label.

```text
data/
  train/
    HELLO/
      image001.jpg
      image002.jpg
    THANK_YOU/
      image001.jpg
  val/
    HELLO/
    THANK_YOU/
  test/
    HELLO/
    THANK_YOU/
```

Use at least two classes for a classification demo. Keep test images out of training and tuning. If images are from multiple signers or recording sessions, split by signer/session where possible to reduce data leakage. Do not rename arbitrary hand images as ISL signs unless the labels are verified.

## 3. Train
```bash
python train.py --data data --epochs 10 --batch-size 32
```
The best validation checkpoint is saved to `artifacts/best_model.pt`; class labels are saved to `artifacts/classes.json`.

## 4. Evaluate on the test set
```bash
python evaluate.py --data data
```
This prints accuracy, a classification report, and a confusion matrix. Save the output for the report. Do not invent or copy example numbers.

## 5. Run the demo
```bash
streamlit run app.py
```
Upload a test image and inspect the predicted class and model confidence. Confidence is not the same as correctness.

## 6. Evidence to collect for Review 1
- Screenshot of the training run and loss/accuracy output.
- Test-set metrics and confusion matrix.
- Screenshot of the working image-upload demo.
- Dataset source, licence, class names, number of images per class, and split strategy.
- A short list of implemented vs planned modules.
