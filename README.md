# CancerCellVision

**CancerCellVision** is a deep learning project focused on detecting and classifying cancer cells from histopathology segmentation images. Leveraging Convolutional Neural Networks (CNNs) and a robust medical imaging dataset, this project aims to aid early cancer diagnosis and support treatment planning through automated and accurate analysis.

## 🧠 Overview

CancerCellVision applies computer vision and deep learning techniques to medical images to identify and distinguish between healthy and cancerous cells. By automating cell classification, it supports pathologists and reduces diagnostic errors.

## 🔍 Features

- 🧬 Automated cancer cell detection from segmentation images  
- 🔎 CNN-based classification architecture  
- 📊 High accuracy with thorough training and validation  
- 📁 Modular codebase for preprocessing, training, evaluation, and inference  
- 📷 Support for histopathology datasets (e.g., [PatchCamelyon](https://github.com/basveeling/pcam), [MoNuSeg](https://monuseg.grand-challenge.org/), etc.)



## 🧪 Technologies Used

- Python 3.x  
- TensorFlow or PyTorch  
- OpenCV  
- FastAPI (for API service)  
- NumPy, Pandas  
- Matplotlib, Seaborn  

## 📦 Dataset

CancerCellVision uses histopathology image datasets such as:
- **PatchCamelyon (PCam)**
- **MoNuSeg**  
Instructions for downloading and placing datasets can be found in `data/README.md`.

## 🚀 Getting Started

Follow these steps to set up and run the project:

```bash
# Clone the repository
git clone https://github.com/AnojAryal/CancerCellVision && \
cd CancerCellVision && \

# Create and activate a virtual environment
python3 -m venv venv && \
source venv/bin/activate && \

# Install dependencies
pip install --upgrade pip && \
pip install -r requirements.txt && \

# Create results directory and a temporary results file
mkdir -p results && \
touch results/temp_results.txt && \

# Run the main cancer detection script
python run/detecty.py

# (Optional) Run the FastAPI service
uvicorn services.main:app --reload --port 8001


## 🤝 Contributing

We welcome contributions to improve CancerCellVision!

Feel free to **open a pull request from the `detection_api` branch to the `develoipment` branch** if you're interested in contributing.  
Bug fixes, feature enhancements, or suggestions are all appreciated.

Please follow the project structure, write clean, well-documented code, and ensure compatibility before submitting.


