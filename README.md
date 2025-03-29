# Resume Screening System

## Overview
The **Resume Screening System** is a machine learning-based web application that categorizes resumes using BERT embeddings and a trained deep learning model. This project utilizes **Flask** for the backend, **TensorFlow** for classification, and **PyTorch** with the **BERT model** for text embedding.

![Resume Screening1](https://github.com/user-attachments/assets/89531fa3-2bde-44a7-98f0-aa3a47963c74)
![Resume Screening 2](https://github.com/user-attachments/assets/7dea010e-edb0-46be-9c67-5e189f74ef4e)



## Features
- Uses **BERT (Bidirectional Encoder Representations from Transformers)** for generating high-quality text embeddings.
- A **TensorFlow-trained model** predicts the category of the resume.
- **Flask-based API** for easy integration.
- **Supports GPU acceleration** for faster inference.
- **REST API endpoint** to predict resume categories.

## Technologies Used
- **Python**
- **Flask**
- **TensorFlow/Keras**
- **PyTorch**
- **Transformers (Hugging Face)**
- **NumPy**
- **Pickle**
- **HTML, CSS, JavaScript (for UI)**

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pip
- Virtual Environment (optional but recommended)
- TensorFlow
- PyTorch
- Hugging Face Transformers

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/abhay-2108/Resume-Screening-system.git
   cd Resume-Screening-system
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Download the pre-trained BERT model (if not cached already):
   ```sh
   from transformers import BertTokenizer, BertModel
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   bert_model = BertModel.from_pretrained("bert-base-uncased")
   ```
5. Place the trained model and encoder in the respective directory:
   ```
   \Resume Screening System
esume_classifier_bert.h5
   \Resume Screening System\label_encoder.pkl
   ```
6. Run the Flask app:
   ```sh
   python app.py
   ```
7. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## API Usage
### Endpoint: `/predict`
#### Method: `POST`
#### Request:
```json
{
  "resume": "Sample resume text here..."
}
```
#### Response:
```json
{
  "predicted_category": "Software Engineer"
}
```

## Troubleshooting
- Ensure **TensorFlow** and **PyTorch** are correctly installed.
- Check that the **BERT model** and **trained classifier** exist in the specified paths.
- If using **GPU**, confirm that CUDA is properly set up.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



