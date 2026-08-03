from setuptools import setup, find_packages

setup(
    name='churn_predictor',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'Flask==2.2.3',
        'Flask-CORS==3.0.10',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'xgboost==1.7.6',
        'joblib==1.2.0',
        'gunicorn==20.1.0',
    ],
)