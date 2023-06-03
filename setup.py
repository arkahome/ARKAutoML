from setuptools import setup
from distutils import util



setup(
    name='arkautoml',
    version='1.0.0',
    description='A test module to check if python library can be installed',
    author='Arka Prava Bandyopadhyay',
    author_email='arkahome@gmail.com',
    package_dir = {
        'arkautoml' : util.convert_path('src/ARKAutoML')
    },
    packages=['arkautoml'],  # List all the packages or modules your library includes
    install_requires= ['fastai==2.4',
                        'tabulate==0.8.9',
                        'mlflow==1.18.0',
                        'scikit-learn==0.24.1',
                        'loguru==0.5.3',
                        'xgboost==1.4.2',
                        'lightgbm==3.2.1',
                        'optuna==2.8.0',
                        'protobuf==3.20'],  # List any external dependencies
)
