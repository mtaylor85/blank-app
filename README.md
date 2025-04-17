SmartCartAI

💡 Project Description
SmartCartAI is a machine learning-powered web application designed to enhance the grocery shopping experience. It helps users plan purchases more efficiently, reduce food waste, and make eco-conscious decisions by forecasting grocery needs and recommending sustainable alternatives. The project was developed as part of the Machine Learning Design course at the University of Cincinnati.
🔧 Technical Design
The system is structured around key components of an MLOps framework:
- Data Ingestion: Batch processing from historical grocery transaction data.
- Data Validation & Preprocessing: Ensures data quality using schema validation and outlier handling.
- Experiment Tracking: Hyperparameters, model performance, and artifacts logged via MLflow.
- Model Development: Trained regression and classification models for prediction and categorization.
- Deployment (Simulated): Mimicked via a FastAPI container, with considerations for future cloud or edge deployment.
🔁 ML Pipeline
- Data Versioning: Implemented via DVC for reproducibility.
- Data Preprocessing: Missing value imputation, encoding, normalization.
- Modeling Framework: Scikit-learn, XGBoost, and Prophet used for different sub-tasks.
- Monitoring Strategy: Suggestions made for alerting, logging, and drift detection in future iterations.
📊 Model Performance
Our final models demonstrated strong generalization ability with key metrics including:
- RMSE: 12.56 (Forecasting)
- Accuracy: 87.3% (Classification)
- MAE: 9.87  
Multiple iterations were conducted to tune performance and assess trade-offs between bias and variance.
🌐 Deployment Strategy
Although we have not deployed the system on a cloud infrastructure, a local deployment was mimicked using FastAPI. In a real-world implementation, this could be extended to:
- Cloud-based deployment (e.g., AWS EC2, Azure App Service)
- On-prem hosting within a grocery retailer's IT environment
- Edge devices like smart fridges or in-store kiosks

🚀 How to Run the Code
1. Clone the repo:
git clone https://github.com/mtaylor85/blank-app.git
cd blank-app

2. Set up your virtual environment:
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Launch the Streamlit app:
streamlit run app.py
👥 Team Members
- Mithisha Brilent Tavares  
- Jyothirmayi Bavi Reddy  
- Mathew Taylor  
- Enock Owusu  
- Yasmine Loussaief

📎 Repository
GitHub Source Code & Documentation: https://github.com/mtaylor85/blank-app
