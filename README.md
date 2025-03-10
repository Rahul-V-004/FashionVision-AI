# Fashion Classification API and GUI - Installation Guide

This guide will help you set up and run both the Django REST API and the Streamlit GUI application for the fashion classification model.

## Prerequisites

- Python 3.8+ installed
- pip (Python package manager)
- PyTorch installed
- Your trained model files:
  - `best_fashion_model.pth`
  - `color_encoder.pkl`
  - `product_type_encoder.pkl`
  - `season_encoder.pkl`
  - `gender_encoder.pkl`

## 1. Set Up Django API

### Create a New Django Project

```bash
# Install required packages
pip install django django-cors-headers torch torchvision pillow matplotlib

# Create a new Django project
django-admin startproject fashion_project
cd fashion_project

# Create a new Django app
python manage.py startapp fashion_api
```

### Configure the Django Project

1. Copy the provided Django code files to their respective locations:
   - `app.py` → `fashion_api/views.py`
   - `django-settings.py` → `fashion_project/settings.py`
   - `django-urls.py` → `fashion_project/urls.py`

2. Create a new file `fashion_api/urls.py` with the following content:
   ```python
   from django.urls import path
   from . import views

   urlpatterns = [
       path('predict/', views.predict, name='predict'),
       path('info/', views.model_info, name='model_info'),
   ]
   ```

3. Copy your model files to the project root directory:
   - `best_fashion_model.pth`
   - `color_encoder.pkl`
   - `product_type_encoder.pkl`
   - `season_encoder.pkl`
   - `gender_encoder.pkl`

### Run the Django API Server

```bash
python manage.py migrate
python manage.py runserver
```

The API should now be running at http://localhost:8000/api/

## 2. Set Up Streamlit GUI

### Create a New Directory for the Streamlit App

```bash
mkdir fashion_streamlit
cd fashion_streamlit
```

### Install Required Packages

```bash
pip install streamlit pandas pillow matplotlib requests
```

### Copy the Streamlit Code

Copy the `streamlit-app.py` file to your fashion_streamlit directory as `app.py`.

### Run the Streamlit App

```bash
streamlit run app.py
```

The Streamlit app should now be running at http://localhost:8501

## Testing the Application

1. Ensure both the Django API and Streamlit app are running
2. Open the Streamlit app in your browser
3. Upload an image of a fashion item or use a sample image
4. The app will send the image to the API for prediction and display the results

## Troubleshooting

### API Connection Issues
- Ensure the API URL in the Streamlit app is correct (default: http://localhost:8000/api)
- Check that CORS is properly configured in the Django API
- Verify that both servers are running

### Model Loading Issues
- Ensure all model files are in the correct locations
- Check console logs for any error messages related to model loading
- Verify that the model architecture in the API matches your training model

### Image Processing Issues
- Ensure uploaded images are in JPG, JPEG, or PNG format
- Check that the image transformations match those used during training
