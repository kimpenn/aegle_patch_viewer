How to install the environment for Aegle Patch Viewer
```
# Create the environment with numpy and pandas, setting conda-forge as priority
conda create --name aegle_patch_viewer -c conda-forge numpy pandas
conda activate aegle_patch_viewer 
# Install streamlit and Pillow
pip install streamlit
brew install libjpeg && pip3 install Pillow
```

Run the application
```
streamlit run app.py --server.headless true
```

Install Git LFS to pull images
```
brew install git-lfs
git lfs install --system
```