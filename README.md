# Create dataset features:
```
python extract_features.py -d dataset/ -o features.h5
```
# Train:
```
python train.py -d features.h5 -m model.pb
```
