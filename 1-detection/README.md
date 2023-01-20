
Datases are provided in ```./images```,  all the original images are in the folder.

The images are divided into training set and test set form the original images according to the bounding box annotations. To divide the images, run the following command:

```python timage_train_val_division.py ```

Next, we need to delete those images without chimpanzees. To clean the images, run the following command:

```python data_clean.py ```

The divided and cleaned images are stored in ```./images/train``` and ```./images/val```.

The trained models are stored in ```./models/trained```




