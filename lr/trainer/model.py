import tensorflow as tf
import multiprocessing

DATA_COLUMNS = ['salience', 'entity_type', 'mentions',
               'noun_type', 'pos_title', 'tag']
DATA_DEFAULTS = [[0.], [0], [0], [0], [0], ['']]
LABEL_COLUMN = 'tag'
LABEL_KEYS = ['yes', 'no']

INPUT_DATA = [
    tf.feature_column.numeric_column('salience'),
    tf.feature_column.numeric_column('entity_type'),
    tf.feature_column.numeric_column('mentions'),
    tf.feature_column.numeric_column('noun_type'),
    tf.feature_column.numeric_column('pos_title'),
]

def create_estimator(config, model_dir):
  (salience, entity_type, mentions, noun_type, pos_title) = INPUT_DATA

  lr_classifier = tf.contrib.learn.LinearClassifier(
      feature_columns=INPUT_DATA,
      model_dir=model_dir
  )
  return lr_classifier

def get_cleaned_colums():
    return set(DATA_COLUMNS) - {col.name for col in INPUT_DATA} - \
        {LABEL_COLUMN}

##
#Parsing and IO methods borrowed from cloudml-samples-master GitHub repo
##

def parse_label_column(label_string_tensor):
  """Parses a string tensor into the label tensor
  Args:
    label_string_tensor: Tensor of dtype string. Result of parsing the
    CSV column specified by LABEL_COLUMN
  Returns:
    A Tensor of the same shape as label_string_tensor, should return
    an int64 Tensor representing the label index for classification tasks,
    and a float32 Tensor representing the value for a regression task.
  """
  # Build a Hash Table inside the graph
  table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABEL_KEYS))

  # Use the hash table to convert string labels to ints and one-hot encode
  return table.lookup(label_string_tensor)


def csv_serving_input_fn():
  """Build the serving inputs."""
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return tf.contrib.learn.InputFnOps(features, None, {'csv_row': csv_row})


# [START serving-function]
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_DATA:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

  features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in inputs.iteritems()
  }
  return tf.contrib.learn.InputFnOps(features, None, inputs)
# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=DATA_DEFAULTS)
  features = dict(zip(DATA_COLUMNS, columns))

  # Remove unused columns
  for col in get_cleaned_colums():
    features.pop(col)
  return features


def input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=0,
                      batch_size=200):
  """Generates features and labels for training or evaluation.
  This uses the input pipeline based approach using file name queue
  to read data so that entire data is not loaded in memory.

  Args:
      filenames: [str] list of CSV files to read data from.
      num_epochs: int how many times through to read the data.
        If None will loop through data indefinitely
      shuffle: bool, whether or not to randomize the order of data.
        Controls randomization of both file order and line order within
        files.
      skip_header_lines: int set to non-zero in order to skip header lines
        in CSV files.
      batch_size: int First dimension size of the Tensors returned by
        input_fn
  Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
  """
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=shuffle)
  reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

  _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

  # Parse the CSV File
  features = parse_csv(rows)

  # This operation builds up a buffer of parsed tensors, so that parsing
  # input data doesn't block training
  # If requested it will also shuffle
  if shuffle:
    features = tf.train.shuffle_batch(
        features,
        batch_size,
        min_after_dequeue=2 * batch_size + 1,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )
  else:
    features = tf.train.batch(
        features,
        batch_size,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )

  return features, parse_label_column(features.pop(LABEL_COLUMN))
