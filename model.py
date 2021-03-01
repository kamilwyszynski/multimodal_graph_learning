import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow.keras.backend as K
from tensorflow.keras import Model, optimizers, losses, metrics

from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression, link_classification
from stellargraph.data import EdgeSplitter

import multiprocessing
from IPython.display import display, HTML
import matplotlib.pyplot as plt

def split_graph(g, p_test=0.1, p_train=0.1):
    # TEST
    edge_splitter_test = EdgeSplitter(g)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    g_test, edges_test, labels_test = edge_splitter_test.train_test_split(
        p=p_test, method="global", edge_label='image2word'
    )

    # TRAIN
    edge_splitter_train = EdgeSplitter(g_test)

    # Sampling for the second time for train to eliminate overlap with test:
    g_train, edges_train, labels_train = edge_splitter_train.train_test_split(
        p=p_train, method="global", edge_label='image2word'
    )

    return edges_train, edges_test, labels_train, labels_test

def get_hinsage_generators(g, edges_train, edges_test, labels_train, labels_test,
                           batch_size=20, num_samples=[8,4]):

    # TODO: explain what exactly is the num_samples for
    generator = HinSAGELinkGenerator(
        g, batch_size, num_samples, head_node_types=["word", "image"]
    )

    train_gen = generator.flow(edges_train, labels_train, shuffle=True)
    test_gen = generator.flow(edges_test, labels_test)

    return generator, train_gen, test_gen

def get_hinsage_model(generator, train_gen, test_gen, num_samples=[8,4], hinsage_layer_sizes=[32, 32], bias=True, dropout=0.0):

    assert len(hinsage_layer_sizes) == len(num_samples)

    hinsage = HinSAGE(
        layer_sizes=hinsage_layer_sizes, generator=generator, bias=bias, dropout=dropout
    )

    # Expose input and output sockets of hinsage:
    x_inp, x_out = hinsage.in_out_tensors()

    # Final estimator layer
    # TODO: change to link_classifier
    score_prediction = link_regression(edge_embedding_method="concat")(x_out)

    def root_mean_square_error(s_true, s_pred):
        return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


    model = Model(inputs=x_inp, outputs=score_prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=1e-2),
        loss=losses.mean_squared_error,
        metrics=[root_mean_square_error, metrics.mae],
    )

    return model

def perform(model, generator, train_gen, test_gen, labels_test, num_workers=4, epochs=20):

    test_metrics = model.evaluate(
        test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
    )

    print("Untrained model's Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        verbose=1,
        shuffle=False,
        use_multiprocessing=False,
        workers=num_workers,
    )

    test_metrics = model.evaluate(
        test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
    )

    print("Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    y_true = labels_test
    # Predict the rankings using the model:
    y_pred = model.predict(test_gen)
    # Mean baseline rankings = mean movie ranking:
    y_pred_baseline = np.full_like(y_pred, np.mean(y_true))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
    mae = mean_absolute_error(y_true, y_pred_baseline)
    print("Mean Baseline Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("\nModel Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)

    h_true = plt.hist(y_true, bins=30, facecolor="green", alpha=0.5)
    h_pred = plt.hist(y_pred, bins=30, facecolor="blue", alpha=0.5)
    plt.xlabel("ranking")
    plt.ylabel("count")
    plt.legend(("True", "Predicted"))
    plt.show()