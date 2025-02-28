
# System for change detection from satellite data using artificial intelligence

Repository for Master Thesis "System for change detection from satellite data using artificial intelligence" at BUT Brno 2025.

## Requirements

### Python 3.13.2 
[link](https://www.python.org/downloads/release/python-3132/)
### Virtual enviroment
create
```bash  
python -m venv 
``` 
activate
```bash  
.\venv\Scripts\activate
```
### PyTorch CUDA
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 
```
### Other libraries
```bash
pip install -r requirements.txt
```
### Run
```bash
python train.py
```
### Evaluate
```bash
python eval.py
```