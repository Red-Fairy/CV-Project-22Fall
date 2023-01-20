## Dataset
To get the images, please download it from https://disk.pku.edu.cn:443/link/36F7BEAB17720475692C32287A5D419A and put it in this folder, all the original images are in it.

To get the labels, please download it from https://disk.pku.edu.cn:443/link/4D22B965DFEBCBC1D0D2B64D9DE65475
 and put it in this folder, all the divided labels are in it.

The images are divided into training set and test set form the original images according to the bounding box annotations. To divide the images, run the following command:

```python timage_train_val_division.py ```

Next, we need to delete those images without chimpanzees. To clean the images, run the following command:

```python data_clean.py ```

The divided and cleaned images are stored in ```./images/train``` and ```./images/val```.


## Detect
To do the detection task, run the following command:

```python detect.py --weights models/trained/yolov5l_40_epochs.pt --source images/val```

The results will be stored in ``./results/detect``.

## Train
To get the models that have not been trained , please download it from https://disk.pku.edu.cn:443/link/C025D5115396E5BAFA113BACBF208B81
 and put it in```./models```.

To train the model, run the following command:

```python train.py --data data/xingxing.yaml --epochs 40 --weights models/not_trained/yolov5l.pt --cfg yolov5l.yaml  --batch-size 128```

The checkpoints will be stored in ``./results/train``.

To get our pretrained checkpoint, please download it from https://disk.pku.edu.cn:443/link/8DD5153EBC3795E97CBD56EB8B8FA2B8
 and put it in```./models```.


## Test
You may run the following command to evaluate the model.

```python val.py  --data data/xingxing.yaml --weights models/trained/yolov5l_40_epochs.pt --augment```

The result will be stored in ``./results/val``.
