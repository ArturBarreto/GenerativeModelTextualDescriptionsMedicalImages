import os
import pathlib
import random
import shutil

base_dir = pathlib.Path("aclImdb")
validation_dir = base_dir / "validation"
train_dir = base_dir / "train"

for category in ("neg", "pos"):
    os.makedirs(validation_dir / category)
    files = os.listdir(train_dir / category)
    # Shuffle the list of training files using a seed, to ensure we get the same
    # validation set every time we run the code
    random.Random(1337).shuffle(files)
    # Take 20% of the training files to use for validation
    num_val_samples = int(0.2 * len(files))
    validation_files = files[-num_val_samples:]
    for file in validation_files:
        # Move the files to aclImdb/val/neg and aclImdb/val/pos
        shutil.move(train_dir / category / str(file),
                    validation_dir / category / str(file))
