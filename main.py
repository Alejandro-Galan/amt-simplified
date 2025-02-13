import os
import random
from itertools import chain

import numpy as np
import tensorflow as tf

from experimentation import k_fold_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 4
    SYMER_THRESHOLD = 30

    # Use the entire training partition
    # Compare performance with and without transfer learning

    # Get baselines
    k_fold_experiment(
        task="amt",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    ##################################################################################
    ## TODO David. After obtaining a functional code. 
    # Consider exploring hyperparameters, more experimentation or changes in the net.
    # Always with the main goal in mind (improving transcription based on data), avoid meaningless tests
    ##################################################################################

 