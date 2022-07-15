# DWT weights

## Development

### Prepare conda environment

```bash
# Create env with Python 3.9
conda create -n dwt-lpips python=3.9
conda activate dwt-lpips

# Install torch related modules
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

### Install requirements.txt

```bash
python -m pip install -r requirements.txt
```
