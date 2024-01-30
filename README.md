
# Estrutura dataset:

```
dataset/
...classe_1/
........image1.png
........image2.png
........imagen.png
...classe_2/
........image1.png
........image2.png
........imagen.png
...classe_n/
x```


# Extract dataset to features:
```
python extract_features.py -d dataset/ -o features.h5
```
# Train:
```
python train.py -d features.h5 -m model.pb
```
