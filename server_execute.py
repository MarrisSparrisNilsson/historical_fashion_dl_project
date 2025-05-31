import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, NASNetMobile, ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

tf.keras.mixed_precision.set_global_policy("mixed_float16")

AUTOTUNE = tf.data.AUTOTUNE

DATA_DIR = str(Path().absolute() / "datasets")

# DATA_DIR = r"C:/Users/maxim/Documents/DLProjectDress/datasets"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
# INPUT_SIZE = (224, 224)

# compute global mean & std of the 'year' label
all_dfs = [pd.read_csv(os.path.join(DATA_DIR, f"fold{i}.csv")) for i in range(10)]
df_all = pd.concat(all_dfs, ignore_index=True)
YEAR_MEAN = df_all["year"].mean()
YEAR_STD = df_all["year"].std()
print(f"Label mean year = {YEAR_MEAN:.2f}, std = {YEAR_STD:.2f}")


def build_model(base, lr=LR):
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(1)(x)  # regression head

    model = Model(base.input, out)
    print(model.summary())
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return model
