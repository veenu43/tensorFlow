
pip install --upgrade virtualenv

# Create new virtual environment
python -m venv --system-site-packages .\venv


# Activate tensor FLow
.\venv\Scripts\activate

# Deactiavte TS Flow
.\venv\Scripts\deactivate

# install tensorflow
pip install --upgrade tensorflow

# verify installation
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"


# 