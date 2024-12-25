from setuptools import setup, find_packages

setup(
    name='ISES',  # Replace with your package name
    version='0.1.0',        # Replace with your package version
    description='blank',  # Add a brief description
    author='Your Name',  # Replace with your name
    packages=find_packages(),
    install_requires=[
        'torch',               # Install PyTorch
        'numpy',        # Install torchvision (if needed)
        'pandas',        # Install torchaudio (if needed)
        # Add any other PyPI packages your project depends on
    ],
    package_data={
        'model': ['model.py'],  # Include data files
        'data': ['test_MPRA.txt', 'train_MPRA.txt', 'trainsolutions.txt', 'JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt']
    },
)