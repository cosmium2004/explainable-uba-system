# Cloud User Behavior Analytics (UBA) System

A comprehensive machine learning-based system for detecting anomalous user behavior in cloud environments using explainable AI techniques.

## Overview

This Cloud UBA system monitors user activities in cloud environments and identifies potentially malicious or anomalous behavior patterns using Random Forest classification. The system provides real-time predictions, batch processing capabilities, and explainable AI insights through SHAP and LIME explanations.

### Key Features

- **Real-time Anomaly Detection**: Instant analysis of user actions with confidence scores
- **Batch Processing**: Analyze historical data in bulk
- **Explainable AI**: SHAP and LIME explanations for model predictions
- **Automated Alerting**: Email notifications for high-severity anomalies
- **Data Drift Detection**: Monitor model performance degradation
- **Web Dashboard**: Interactive UI for predictions and monitoring
- **RESTful API**: Easy integration with existing systems
- **Performance Tracking**: Comprehensive metrics and statistics

## Architecture

```
Cloud UBA System
├── Data Layer (SQLite)
├── ML Models (Random Forest + Preprocessor)
├── API Layer (Flask REST API)
├── Monitoring & Alerting
├── Explainability (SHAP/LIME)
└── Web Interface
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (venv or virtualenv)
- 2GB+ RAM recommended
- Windows/Linux/macOS

## Installation

### 1. Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd <project-directory>

# Or download and extract the ZIP file
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
DEBUG=True
ALERT_THRESHOLD=0.7
EMAIL_NOTIFICATIONS_ENABLED=False
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-password
SENDER_EMAIL=alerts@example.com
RECIPIENT_EMAILS=admin@example.com,security@example.com
```

**Security Note**: Generate secure keys for production:
```python
import secrets
print(secrets.token_hex(32))  # For SECRET_KEY
print(secrets.token_hex(16))  # For API_KEY
```

### 5. Initialize Database

```bash
python init_db.py
```

This creates the SQLite database with required tables and sample data.

### 6. Verify Model Files

Ensure these files exist in the `models/` directory:
- `random_forest_tuned.pkl` - Trained Random Forest model
- `preprocessor.pkl` - Feature preprocessing pipeline

If missing, you'll need to train the model using the Jupyter notebook.

## Training the Model (Optional)

If you need to train or retrain the model:

### 1. Install Jupyter

```bash
pip install jupyter notebook
```

### 2. Open the Training Notebook

```bash
jupyter notebook Cloud_UBA_Setup.ipynb
```

### 3. Run All Cells

The notebook will:
- Generate synthetic cloud user behavior data
- Train a Random Forest classifier
- Perform hyperparameter tuning
- Save the trained model and preprocessor
- Generate evaluation metrics and visualizations

### 4. Model Files Location

After training, models are saved to:
- `models/random_forest_tuned.pkl`
- `models/preprocessor.pkl`

## Running the Application

### Development Mode

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production Mode (Using Deployer)

```bash
cd Deployer
python app.py
```

Or use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

## Usage Guide

### Web Interface

1. **Home Page**: `http://localhost:5000/`
   - Overview of the system

2. **Single Prediction**: `http://localhost:5000/predict`
   - Test individual user events
   - Get real-time anomaly predictions
   - View explanations

3. **Batch Processing**: `http://localhost:5000/batch`
   - Upload CSV files with multiple events
   - Process historical data
   - Download results

4. **Dashboard**: `http://localhost:5000/dashboard`
   - View system statistics
   - Monitor alerts
   - Track performance metrics

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:5000/health
```

#### 2. Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "user_id": "user-001",
    "timestamp": "2024-01-15T14:30:00Z",
    "action_type": "download",
    "resource_id": "file-001",
    "ip_address": "192.168.1.100",
    "data_volume_mb": 150.5,
    "location": "New York",
    "device_type": "desktop",
    "failed_attempts": 0,
    "location_change": false,
    "explain": true
  }'
```

#### 3. Batch Prediction
```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "instances": [
      {
        "user_id": "user-001",
        "timestamp": "2024-01-15T14:30:00Z",
        "action_type": "login",
        "ip_address": "192.168.1.100",
        "device_type": "desktop"
      },
      {
        "user_id": "user-002",
        "timestamp": "2024-01-15T15:00:00Z",
        "action_type": "admin",
        "ip_address": "10.0.0.1",
        "device_type": "mobile",
        "failed_attempts": 5
      }
    ],
    "explain": false
  }'
```

#### 4. Get Model Information
```bash
curl -X GET http://localhost:5000/api/model \
  -H "X-API-Key: your-api-key-here"
```

#### 5. Get Performance Metrics
```bash
curl -X GET "http://localhost:5000/api/metrics?time_window=3600" \
  -H "X-API-Key: your-api-key-here"
```

#### 6. Get System Statistics
```bash
curl -X GET "http://localhost:5000/api/statistics?days=30" \
  -H "X-API-Key: your-api-key-here"
```

### Input Data Format

#### Required Fields
- `user_id` (string): Unique user identifier
- `timestamp` (ISO 8601 string): Event timestamp
- `action_type` (string): Type of action (login, logout, download, upload, delete, view, admin)

#### Optional Fields
- `resource_id` (string): Resource being accessed
- `ip_address` (string): User's IP address
- `data_volume_mb` (float): Data transfer volume in MB
- `location` (string): Geographic location
- `device_type` (string): Device type (desktop, mobile, tablet, server)
- `failed_attempts` (integer): Number of failed login attempts
- `location_change` (boolean): Whether location changed from previous access
- `explain` (boolean): Request explanation for prediction

### Response Format

```json
{
  "success": true,
  "prediction": {
    "user_id": "user-001",
    "timestamp": "2024-01-15T14:30:00Z",
    "is_anomaly": true,
    "anomaly_score": 0.85,
    "confidence": 0.85,
    "prediction_time": "2024-01-15T14:30:05Z",
    "explanation": {
      "text_explanation": "This event was flagged as anomalous...",
      "shap_explanation": {...},
      "lime_explanation": {...}
    }
  }
}
```

## Testing

### Generate Test Data

```bash
python generate_test_batch.py
```

This creates `test_batch.csv` with 50 sample events.

### Run Tests

```bash
pytest
```

### Manual Testing

Use the web interface at `http://localhost:5000/predict` to test individual predictions with various scenarios:

1. **Normal Behavior**: Regular login during business hours
2. **Suspicious Behavior**: Admin action at 3 AM with failed attempts
3. **Data Exfiltration**: Large download from unusual location

## Configuration

### Alert Thresholds

Modify in `.env`:
```env
ALERT_THRESHOLD=0.7  # Anomaly score threshold (0.0-1.0)
```

### Email Notifications

Enable email alerts:
```env
EMAIL_NOTIFICATIONS_ENABLED=True
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SENDER_EMAIL=alerts@yourcompany.com
RECIPIENT_EMAILS=security@yourcompany.com,admin@yourcompany.com
```

### Database Configuration

Default: SQLite (`instance/cloud_uba.db`)

To use a different database:
```env
DATABASE_URI=sqlite:///path/to/database.db
```

## Project Structure

```
cloud-uba/
├── api/                      # API routes and middleware
│   ├── routes.py            # REST API endpoints
│   ├── middleware.py        # Authentication & validation
│   └── validators.py        # Input validation schemas
├── database/                 # Database layer
│   ├── connector.py         # Database connection handling
│   ├── models.py            # Database schema definitions
│   └── queries.py           # Common database queries
├── models/                   # ML models and prediction logic
│   ├── model_loader.py      # Model loading and caching
│   ├── prediction.py        # Prediction functions
│   ├── random_forest_tuned.pkl  # Trained model
│   └── preprocessor.pkl     # Feature preprocessor
├── explainers/              # Explainable AI components
│   ├── shap_explainer.py    # SHAP explanations
│   ├── lime_explainer.py    # LIME explanations
│   └── report_generator.py  # Explanation report generation
├── monitoring/              # Monitoring and alerting
│   ├── alerting.py          # Alert generation and notification
│   ├── data_drift.py        # Data drift detection
│   ├── logger.py            # Logging configuration
│   └── performance_tracker.py  # Performance metrics
├── utils/                   # Utility functions
│   └── preprocessing.py     # Feature extraction
├── templates/               # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── batch.html
│   └── dashboard.html
├── static/                  # Static assets (CSS, JS)
├── data/                    # Data files
│   ├── raw/                 # Raw data
│   └── processed/           # Processed features
├── logs/                    # Application logs
├── instance/                # Instance-specific files
│   └── cloud_uba.db        # SQLite database
├── app.py                   # Main Flask application
├── config.py                # Configuration settings
├── init_db.py              # Database initialization
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── Cloud_UBA_Setup.ipynb   # Model training notebook
```

## Monitoring and Maintenance

### View Logs

```bash
# Application logs
tail -f logs/cloud_uba.log

# Alert logs
tail -f logs/alerts.jsonl
```

### Database Maintenance

```bash
# Backup database
cp instance/cloud_uba.db instance/cloud_uba_backup_$(date +%Y%m%d).db

# View database
sqlite3 instance/cloud_uba.db
```

### Performance Monitoring

Access metrics via API:
```bash
curl -X GET http://localhost:5000/api/metrics \
  -H "X-API-Key: your-api-key-here"
```

## Troubleshooting

### Issue: Model file not found

**Solution**: Ensure model files exist in `models/` directory or train the model using the Jupyter notebook.

### Issue: Database connection error

**Solution**: Run `python init_db.py` to initialize the database.

### Issue: Import errors

**Solution**: Activate virtual environment and reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Issue: API returns 401 Unauthorized

**Solution**: Include valid API key in request header:
```bash
-H "X-API-Key: your-api-key-here"
```

### Issue: Email notifications not working

**Solution**: 
1. Verify SMTP settings in `.env`
2. For Gmail, use App Password instead of regular password
3. Check firewall settings for SMTP port

## Security Considerations

1. **API Keys**: Change default API keys in production
2. **Secret Keys**: Use strong, randomly generated secret keys
3. **HTTPS**: Use HTTPS in production (configure reverse proxy)
4. **Database**: Use PostgreSQL/MySQL for production instead of SQLite
5. **Input Validation**: All inputs are validated before processing
6. **Rate Limiting**: Consider adding rate limiting for production
7. **Credentials**: Never commit `.env` file to version control

## Performance Optimization

### For Large-Scale Deployments

1. **Use Production Database**: PostgreSQL or MySQL
2. **Add Caching**: Redis for model and feature caching
3. **Load Balancing**: Multiple application instances behind load balancer
4. **Async Processing**: Celery for batch processing
5. **Model Optimization**: Quantize model or use lighter algorithms

### Scaling Recommendations

- **< 1000 requests/day**: Single instance with SQLite
- **1000-10000 requests/day**: Multiple instances with PostgreSQL
- **> 10000 requests/day**: Kubernetes cluster with Redis caching

## Dependencies

### Core Dependencies
- Flask 2.2.3 - Web framework
- pandas 1.5.3 - Data manipulation
- numpy 1.24.3 - Numerical computing
- scikit-learn 1.2.2 - Machine learning
- joblib 1.2.0 - Model serialization

### Explainability
- lime 0.2.0.1 - Local interpretable explanations
- SHAP (optional) - SHapley Additive exPlanations

### Monitoring
- python-dotenv 1.0.0 - Environment variable management
- matplotlib 3.7.1 - Visualization
- seaborn 0.12.2 - Statistical visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- Create an issue in the repository
- Contact: [your-email@example.com]

## Acknowledgments

- Built with Flask and scikit-learn
- Explainability powered by LIME and SHAP
- Inspired by modern cloud security practices

---

**Version**: 1.0.0  
**Last Updated**: February 2024  
**Status**: Production Ready
