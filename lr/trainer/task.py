import argparse
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

#Returns an experiment function
def create_experiment(**experiment_args):

  #Creates an experiment using config args and hparams
  def _experiment(run_config, hparams):
    train_input = lambda: model.input_fn(
        hparams.train_files,
        batch_size=hparams.train_batch_size,
    )
    eval_input = lambda: model.input_fn(
        hparams.eval_files,
        batch_size=hparams.eval_batch_size,
    )
    experiment = tf.contrib.learn.Experiment(
        model.create_estimator(
            config=run_config,
            model_dir=args.job_dir
        ),
        train_input_fn=train_input,
        eval_input_fn=eval_input,
        **experiment_args
    )
    return experiment
  return _experiment


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--train-files',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=32
  )
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=32
  )
  parser.add_argument(
      '--eval-files',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      required=False
  )
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--eval-delay-secs',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min-eval-frequency',
      default=None,
      type=int
  )
  parser.add_argument(
      '--train-steps',
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      default=100,
      type=int
  )
  parser.add_argument(
      '--export-format',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON'
  )

  args = parser.parse_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)

  learn_runner.run(
      create_experiment(
          min_eval_frequency=args.min_eval_frequency,
          eval_delay_secs=args.eval_delay_secs,
          train_steps=args.train_steps,
          eval_steps=args.eval_steps,
          export_strategies=[saved_model_export_utils.make_export_strategy(
              model.SERVING_FUNCTIONS[args.export_format],
              exports_to_keep=1,
              default_output_alternative_key=None,
          )]
      ),
      run_config=run_config.RunConfig(model_dir=args.job_dir),
      hparams=hparam.HParams(**args.__dict__)
  )
