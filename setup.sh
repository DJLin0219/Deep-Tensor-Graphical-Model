# Create conda environment
conda create -n detect python=3.10 -y
conda activate detect

# Install Jupyter notebook kernel
conda install -c conda-forge notebook ipykernel -y
python -m ipykernel install --user --name detect

# Core dependencies
conda install numpy scipy pandas networkx matplotlib scikit-learn -y

# PyTorch (auto-select CUDA if available)
# You can manually set CUDA version if needed (e.g. 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Visualization
pip install pyvis

# Export environment file for reproducibility
conda env export > environment.yml