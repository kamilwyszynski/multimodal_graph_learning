import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
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
    edge_splitter_train = EdgeSplitter(g_test, g)

    # Sampling for the second time for train to eliminate overlap with test:
    g_train, edges_train, labels_train = edge_splitter_train.train_test_split(
        p=p_train, method="global", edge_label='image2word'
    )

    return edges_train, edges_test, labels_train, labels_test

# TODO: For some rason, the edge switches from img -> wrd to wrd -> img from time to time
def get_hinsage_generators(g, edges_train, edges_test, labels_train, labels_test,
                           batch_size=20, num_samples=[8,4], shuffle=True, head_node_types=["image", "word"]):

    generator = HinSAGELinkGenerator(
        g, batch_size, num_samples, head_node_types=head_node_types
    )

    train_gen = generator.flow(edges_train, labels_train, shuffle=shuffle)
    test_gen = generator.flow(edges_test, labels_test)

    return generator, train_gen, test_gen

def get_hinsage_model(generator, train_gen, test_gen, num_samples=[8,4], hinsage_layer_sizes=[32, 32], bias=True, dropout=0.0, lr=1e-2, edge_embedding_method='concat', output_act='sigmoid'):

    assert len(hinsage_layer_sizes) == len(num_samples)

    hinsage = HinSAGE(
        layer_sizes=hinsage_layer_sizes, generator=generator, bias=bias, dropout=dropout
    )

    # Expose input and output sockets of hinsage:
    x_inp, x_out = hinsage.in_out_tensors()

    # Final estimator layer
    score_prediction = link_classification(output_dim=1, output_act='sigmoid', edge_embedding_method=edge_embedding_method)(x_out)

    def root_mean_square_error(s_true, s_pred):
        return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


    model = Model(inputs=x_inp, outputs=score_prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        # loss=losses.mean_squared_error,
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
        # metrics=[root_mean_square_error, metrics.mae, 'acc'],
    )

    return model


def perform(model, generator, train_gen, test_gen, labels_test, num_workers=4, epochs=20, verbose=1, shuffle=False):

    test_metrics = model.evaluate(
        test_gen, verbose=verbose, use_multiprocessing=False, workers=num_workers
    )

    print("Untrained model's Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        use_multiprocessing=False,
        workers=num_workers,
    )

    plot_history(history)

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
    # acc = binary_accuracy(y_true, y_pred_baseline)
    print("Mean Baseline Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)
    # print("\taccuracy = ", acc)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # acc = binary_accuracy(y_true, y_pred)
    print("\nModel Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)
    # print("\taccuracy = ", acc)

    h_true = plt.hist(y_true, bins=30, facecolor="green", alpha=0.5)
    h_pred = plt.hist(y_pred, bins=30, facecolor="blue", alpha=0.5)
    plt.xlabel("ranking")
    plt.ylabel("count")
    plt.legend(("True", "Predicted"))
    plt.show()

def plot_history(history):
        metrics = sorted(history.history.keys())
        metrics = metrics[:len(metrics)//2]
        for m in metrics:        
            plt.plot(history.history[m])
            plt.plot(history.history['val_' + m])
            plt.title(m)
            plt.ylabel(m)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.show()