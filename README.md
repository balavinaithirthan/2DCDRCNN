# 2DCDRCNN: Cancer Drug Response Prediction using CNN and Gene Expression Data


**2DCDRCNN** is a deep learning model that utilizes Convolutional Neural Networks (CNNs) to predict the IC50 value of a specific drug on a given cancer cell line. This model takes advantage of one-hot encoded drug data and t-SNE transformed gene expression data to make accurate predictions about the effectiveness of drug-cancer interactions. By learning from known interactions between drugs and cancer cell lines, the project aims to enhance drug development and tailor treatments for individual patients.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

One element that contributes to the complexity and challenges of cancer treatment is the variation in patient responses based on their unique genetic and epigenetic make up. The **2DCDRCNN** project addresses this issue by leveraging molecular structures of drugs and gene expression profiles of cancer cell lines to predict the effectiveness of new drug-cancer combinations. This predictive model can help researchers and clinicians identify potential candidates for testing and tailor treatments for individual patients.

## Features

- Predict Drug-Cancer Interaction: Predict the IC50 value of a specific drug on a given cancer cell line.
- Utilize Gene Expression Data: Utilize t-SNE transformed gene expression data for accurate predictions.
- Enhance Drug Development: Test approved drugs and candidates for their estimated effectiveness on different cancer cell lines.
- Personalized Treatment: Predict patient responses to various cancer drugs using gene expression profiles.

## How It Works

1. **Data Preparation**: The model requires one-hot encoded drug data and t-SNE transformed gene expression data as input.

2. **Convolutional Neural Network (CNN)**: The CNN architecture is used to learn patterns and relationships between drugs and gene expression profiles.

3. **Prediction**: Given a drug and gene expression data, the trained model predicts the IC50 value of the drug on the specific cancer cell line.

4. **Enhanced Drug Testing**: The model's predictions can be used to identify potential drug candidates for testing and streamline drug development processes.

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/2DCDRCNN.git
   cd 2DCDRCNN
   ```

2. Set up your development environment with the required dependencies.

## Usage

1. Prepare your one-hot encoded drug data and t-SNE transformed gene expression data. Mount to your respective google drive

2. Run the model using your preferred Python interpreter:

   ```bash
   python Model.py
   ```

## Example

Predicting IC50 value for a drug-cancer combination:

```plaintext
Enter one-hot encoded drug data: [0, 1, 0, ...]
Enter t-SNE transformed gene expression data: [0.123, 0.456, ...]

Predicting IC50 value...
Estimated IC50 value: 5.67
```

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

---

For any questions or suggestions, please contact [your name/email here]. Happy predicting!
