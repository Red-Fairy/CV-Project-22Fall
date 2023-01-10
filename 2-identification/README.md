## Dataset
Dataset are provided in ```./data```, the images are cropped form the original images according to the bounding box annotations.

## Training
To train the model, run the following command:

```python train_identification.py --experiment exp_path```

Checkpoints and loggers will be stored in ``./checkpoints/exp_path``.

## Testing
You may run the following command to evaluate the model.

```python evaluate.py --pretrained_weights path_to_weights```

We provide our pretrained checkpoint in ``./pretrained_weights/model.pt``, you may also use your own checkpoint.


