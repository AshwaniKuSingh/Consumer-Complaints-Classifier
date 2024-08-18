# Consumer-Complaints-Classifier

Here's a comprehensive README.md file tailored for your "Consumer-Complaints-Classifier" repository, based on the files you provided:

---

# Consumer-Complaints-Classifier

This repository contains a tutorial and implementation for classifying consumer complaints about financial products using various machine learning models, including DistilBERT and RandomForest. The goal is to build and evaluate models that can categorize future complaints into predefined product categories.

## About the Dataset

### Context
The dataset comprises real-world complaints about financial products and services. Each complaint is labeled with a specific product category, making this a supervised text classification problem. The aim is to classify future complaints based on their content using various machine learning algorithms.

### Content
The dataset contains information about complaints made by customers regarding various financial products and services, including Credit Reports, Student Loans, Money Transfers, and more. The complaints span from November 2011 to May 2019.

### Acknowledgements
This dataset is public and was sourced from the U.S. Government's data portal on May 13, 2019. It is considered a U.S. Government Work and can be accessed [here](https://catalog.data.gov/dataset/consumer-complaint-database).

### Inspiration
This project serves as a tutorial for beginners interested in Natural Language Processing (NLP) and supervised text classification.

## Repository Structure

- **`distilbert-model-code-multiclass.ipynb`**: A Jupyter notebook implementing a DistilBERT model for multiclass classification.
- **`RandomForest-Explain-multiclass.ipynb`**: A Jupyter notebook implementing a RandomForest model for multiclass classification, with a focus on model interpretability.
- **`gui-test.py`**: A Python script that provides a GUI for interacting with the trained models using PySimpleGUI.

## Installation

To run the code in this repository, install the following dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy PySimpleGUI nltk lime
```

For NLP tasks, you may also need to download specific NLTK data files:

```python
import nltk
nltk.download('stopwords')
```

## Usage

### DistilBERT Model

The DistilBERT model is implemented in the `distilbert-model-code-multiclass.ipynb` notebook. This notebook covers:

1. **Data Loading and Preprocessing**: 
   - Load and preprocess the dataset, including tokenization and stopword removal.
2. **Model Setup**:
   - Use the `transformers` library to load the DistilBERT model and tokenizer.
3. **Training**:
   - Configure training parameters and train the model using the `Trainer` class.
4. **Evaluation**:
   - Evaluate model performance on the test dataset.

### RandomForest Model

The RandomForest model is detailed in the `RandomForest-Explain-multiclass.ipynb` notebook, covering:

1. **Data Loading and Preprocessing**:
   - Load and preprocess features from the dataset.
2. **Model Training**:
   - Train a RandomForest classifier on the processed data.
3. **Model Explanation**:
   - Use LIME (Local Interpretable Model-agnostic Explanations) to explain the model's predictions and understand feature importance.

### GUI for Model Interaction

The `gui-test.py` script provides a simple GUI for interacting with the trained models. Users can input data through the GUI, and the model's predictions will be displayed. This script uses PySimpleGUI to create the interface and integrates with the trained models for real-time predictions.

To run the GUI:

```bash
python gui-test.py
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides a detailed overview of your project, including the dataset, model implementation, and usage instructions. Let me know if you need any further adjustments!
