import json
import multiprocessing
import multiprocessing.sharedctypes
import numpy
import pathlib
import subprocess
import tensorflow as tf
import pandas as pd


train_dataset = pd.read_csv("/mnt/SOFT_500G/50-train-felix.csv")
test_dataset = pd.read_csv("/mnt/SOFT_500G/25-test-felix.csv")

train_labels = train_dataset.pop('TTA')
test_labels = test_dataset.pop('TTA')


def run(payload):
    genome = payload['genome']
    configuration = payload['configuration']
    run_result = {
        'compiledNN_result': 500.0,
        'primary_metric': 0,
        'training_progression': [],
    }
    tmp_model_path = pathlib.Path('tmp_' + payload['id'] + '.hdf5')
    print('start evaluation of', payload['id'])
    try:
        metrics = []

        for metric in configuration['metrics']:
            if metric[0] == 'f' and metric[-5:] == 'score':
                pass
            else:
                metrics.append(metric)

        model = build_model(genome, configuration)
        model.compile(
            optimizer=genome['optimizer'],
            loss=configuration['loss'],
            metrics=metrics,
        )
        print("Model:")
        model.summary()

        model.optimizer.lr.assign(genome['initial_learning_rate'])

        tf_train_result = model.fit(
            train_dataset.values,
            train_labels.values,
            epochs=genome['training_epochs'])

        run_result['training_progression'] = [
            {
                name: tf_train_result.history[name][ep]
                for name in tf_train_result.history.keys()
            }
            for ep in range(genome['training_epochs'])
        ]

        epoch = 0
        for _ in run_result['training_progression']:
            run_result['training_progression'][epoch]['epoch'] = epoch + 1
            epoch += 1
        print("Train Datasets:")
        print(train_dataset.values)
        print(train_labels.values)
        print("Test Datasets:")
        print(test_dataset.values)
        print(test_labels.values)
        print("eval call")
        print(model.evaluate(x=test_dataset.values, y=test_labels.values))
        evaluate_result = {
            'loss': model.evaluate(x=test_dataset.values, y=test_labels.values)
        }

        for key in evaluate_result:
            run_result[key] = evaluate_result[key]

        tf.keras.models.save_model(
            model,
            str(tmp_model_path),
            save_format='h5')

        run_result['compiledNN_result'] = 0
        print("run result")
        print(run_result)
        print("evaluate result")
        print(evaluate_result)

    except Exception as e:
        print(e)
        run_result['exception'] = str(e)

    # tmp_model_path.unlink()
    tf.keras.backend.clear_session()
    return run_result



def build_dense_layers(genome):
    '''Build sequential layer list of dense layers'''

    layers = []
    for layer in genome['dense_layers']:
        layers.append(tf.keras.layers.Dense(
            layer['units'],
            activation=layer['activation_function'],
        ))

        # if layer['batch_normalization']:
        #     layers.append(tf.keras.layers.BatchNormalization())

        # if layer['drop_out_rate'] > 0.01:
        #     layers.append(tf.keras.layers.Dropout(
        #         rate=layer['drop_out_rate'],
        #     ))

    return layers


def build_layers(genome, configuration):
    '''Build sequential layer list'''

    # input layer
    layers = [
        tf.keras.Input(
            shape=(4, ),
        )
    ]

    # convolution layers
    # layers += build_convolution_layers(genome)

    # flatten between convolutions and denses
    # layers.append(tf.keras.layers.Flatten())

    # dense layers
    layers += build_dense_layers(genome)

    # final layer
    layers.append(tf.keras.layers.Dense(
        configuration['final_layer_neurons'],
        activation=genome['final_layer_activation_function'],
    ))

    if genome['final_layer_batch_normalization']:
        layers.append(tf.keras.layers.BatchNormalization())

    return layers


def build_model(genome, configuration):
    '''Build sequential model'''

    return tf.keras.Sequential(
        layers=build_layers(genome, configuration),
    )
