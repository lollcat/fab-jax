pip install --upgrade pip setuptools wheel
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install dm-acme[jax,tensorflow]
pip install pandas ml_collections joblib wandb tqdm matplotlib jupyterlab hydra-core hydra-joblib-launcher joblib


