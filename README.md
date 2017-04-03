<<<<<<< HEAD
=======
# inception
Using inception V3 model to retrain a place recognition dataset.
>>>>>>> f4b6322155248a2db35c7380c09821b48ee70790
## 1 How to Construct a New Dataset for Retraining

```shell
# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=$HOME/my-custom-data/

# build the preprocessing script.
bazel build inception/build_image_data

# convert the data.
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8
```

```shell
bazel-bin/inception/build_image_data \
  --train_directory=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/data/place_data/train \
  --validation_directory=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/data/place_data/validation \
  --output_directory=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/data \
  --labels_file=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/data/place_data/label.txt \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8
```

After running this script produces files that look like the following:

```shell
  $TRAIN_DIR/train-00000-of-00024
  $TRAIN_DIR/train-00001-of-00024
  ...
  $TRAIN_DIR/train-00023-of-00024

and

  $VALIDATION_DIR/validation-00000-of-00008
  $VALIDATION_DIR/validation-00001-of-00008
  ...
  $VALIDATION_DIR/validation-00007-of-00008
```

where 24 and 8 are the number of shards specified for each dataset,
respectively. Generally speaking, we aim for selecting the number of shards such
that roughly 1024 images reside in each shard. Once this data set is built, you
are ready to train or fine-tune an Inception model on this data set.

## 2 How to Retrain a Trained Model on the new Data
```shell
# Build the model.
bazel build inception/flowers_train

# Path to the downloaded Inception-v3 model.
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"

# Directory where the flowers data resides.
FLOWERS_DATA_DIR=/tmp/flowers-data/

# Directory where to save the checkpoint and events files.
TRAIN_DIR=/tmp/flowers_train/

# Run the fine-tuning on the flowers data set starting from the pre-trained
# Imagenet-v3 model.
```shell
bazel-bin/inception/flowers_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
```

For instance:
```shell
bazel-bin/inception/flowers_train \
  --train_dir=home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/place_train \
  --data_dir=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/data \
  --pretrained_model_checkpoint_path=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/model/inception-v3/model.ckpt-157585 \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
```

## 3 Evaluate the fine-tuned model on a hold-out of the flower data set.

```shell
# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
bazel build inception/flowers_eval

# Directory where we saved the fine-tuned checkpoint and events files.
TRAIN_DIR=/tmp/flowers_train/

# Directory where the flowers data resides.
FLOWERS_DATA_DIR=/tmp/flowers-data/

# Directory where to save the evaluation events files.
EVAL_DIR=/tmp/flowers_eval/

# Evaluate the fine-tuned model on a hold-out of the flower data set.
bazel-bin/inception/flowers_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once
```

For instance:
```shell
bazel-bin/inception/flowers_eval \
  --eval_dir=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/place_eval \
  --data_dir=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/data \
  --subset=validation \
  --num_examples=672 \
  --checkpoint_dir=/home/frank/last_ubuntu_download/models-update-models-1.0/inception/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/place_train \
  --input_queue_memory_factor=1 \
  --run_once
```
