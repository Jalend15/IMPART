## IMPART ##
 Image Matching and Pairing with Adapted RA-CLIP Technology

**Abstract**

Our project seeks to enhance the RA-CLIP model, which utilizes RAM with CLIP for efficient image-text pairing, by adapting it to handle noisier datasets such as blurry images or non-systematic datasets like textbook diagrams. We aim to tailor our querying dataset based on specific domains and assess performance using standard metrics alongside novel downstream tasks. For instance, our model could identify relevant video frames from CCTV footage of a car or match diagrams to those found in textbooks. Initial evaluations will establish benchmarks, with subsequent improvements documented as project milestones.

**Set Up**
```
conda create -n impart python=3.11 
conda install --yes -c pytorch pytorch torchvision cpuonly
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**Scripts**

Go to the root directory and add it to the python path
```
export PYTHONPATH=".:$PYTHONPATH" 
```

Run this to download yfcc dataset to local and split it into train and reference set
```
python scripts/download_and_prepare_dataset.py
```

Run this to load reference embeddings
```
python scripts/load_reference_set.py 
```

Run this to train the model
```
python scripts/train.py
```