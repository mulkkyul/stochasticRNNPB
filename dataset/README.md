# Dataset Processing

These Python scripts prepare datasets for training and recognition tasks. The original dataset is downloaded from the [REBL-Pepper dataset repository](https://github.com/minamar/rebl-pepper-data).

<br><br>

## Prerequisites

1. Download the original dataset [from the official repository](https://github.com/minamar/rebl-pepper-data).  
2. From the `original_animations` directory in the original dataset, copy the following files into the `rebl-pepper` directory:  
   - `data_augmented.csv`: Contains the raw sequence data.  
   - `labels_augmented.csv`: Contains sequence labels.  

<br><br>

## Usage

### 1. Creating Training Data

Run `write_train_set.py` to process the dataset and generate individual training `.csv` files.

```bash
python write_train_set.py
```

**Input:**
- `data_augmented.csv`: Sequence data with joint positions.
- `labels_augmented.csv`: Corresponding sequence labels.

**Output:**
- Individual `.csv` files saved in the `train/` directory.
- A `column_names.txt` file describing the joint names.

<br><br>

### 2. Creating Recognition Data

Run `write_recognition_set.py` to generate a modified version of the dataset for the recognition task. This script applies Principal Component Analysis (PCA) to selected files in the `train` directory, introducing smooth, low-frequency noise along with slight random scaling and shifting. These changes simulate realistic variations and perturbations in the data. The noisy sequences are then reconstructed and saved as new files with a `_noisy` suffix in the `recognition` directory. 

```bash
python write_recognition_set.py
```

**Input:**
- `.csv` files in the `train/` directory.

**Output:**
- Augmented `.csv` files saved in the `recognition/` directory with the `_noisy.csv` suffix.

