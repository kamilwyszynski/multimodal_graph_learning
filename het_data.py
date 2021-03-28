import itertools
import os
import pickle as pkl
import numpy as np
import json
import pandas as pd

import torch
from torch_geometric.data import Data, DataLoader

from img2vec import Img2Vec
from dataset import VizWiz

import gensim.downloader as gensim_api
from gensim.models import KeyedVectors

import networkx as nx
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter

class HetGraph():

    def __init__(self, english_stopwords_path='res/stopwords/english', word2vec_path='models/word2vec.model'):
        self.g = nx.Graph()

        self.x_img = torch.Tensor()
        self.x_wrd = torch.Tensor()
        self.y_img = []
        self.y_wrd = []
        self.edge_index_img = torch.Tensor()
        self.edge_index_wrd = torch.Tensor()
        self.edge_index_i2w = torch.Tensor()
        self.edge_attr_img = torch.Tensor()
        self.edge_attr_wrd = torch.Tensor()
        self.img2vec = Img2Vec()

        if os.path.exists(word2vec_path):
            # self.word2vec = gensim.models.Word2Vec.load(word2vec_path)
            self.word2vec = KeyedVectors.load(word2vec_path)
        else:
            self.word2vec = gensim_api.load('glove-wiki-gigaword-300')
            self.word2vec.save(word2vec_path)

        with open(english_stopwords_path, 'r') as file:
            self.stopwords = file.read().split('\n') # stopwords in english

    def get_details(self):
        # print class details
        print(hg.x_img)
        print(hg.x_wrd)
        print(hg.y_img)
        print(hg.y_wrd)
        print(hg.edge_index_i2w)
        print(hg.edge_attr_img)
        print(hg.edge_attr_wrd)        

    # TODO: Change the process to directly build a networkx
    def add_node(self, img, cap, img_name, fully_connected=True, image_threshold=0.6, word_threshold=0.6):
        img_vec = self.img2vec.get_vector(img).numpy() # calculate image vector
        num_nodes = self.x_img.size()[0] # get number of nodes

        # add image node and its vector as its feature
        self.g.add_node(img_name, label='image', feature=img_vec)   
        self.y_img.append(img_name) # Adding for easier iteration

        for node in self.y_img[:-1]:
            img_sim = self.img2vec.get_cos_sim_np(self.g.nodes[node]['feature'], img_vec) # calculate cos similarity between all saved vectors

            if fully_connected or float(img_sim) > image_threshold:
                self.g.add_edge(node, img_name, weight=float(img_sim), label='image2image')


        for word in list(set(cap.split())):
            word = word.lower()

            to_remove = ['.', ',', '(', ')']
            for r in to_remove:
                word = word.replace(r, '')

            if (word in self.stopwords) or (word not in self.word2vec.vocab.keys()):
                continue # skip if a stopword

            if word not in self.y_wrd:
                word_vec = self.word2vec.get_vector(word) # calculate word vector

                self.g.add_node(word, label='word', feature=word_vec) 
                self.y_wrd.append(word)

                for node in self.y_wrd[:-1]:
                    # TODO: calculate sim directly from vecs
                    wrd_sim = self.word2vec.similarity(node, word) # calculate cos similarity between all saved vectors

                    if fully_connected or float(wrd_sim) > word_threshold:
                        self.g.add_edge(node, word, weight=float(wrd_sim), label='word2word')

            # add word to image edges (not directional)
            self.g.add_edge(img_name, word, label='image2word')

    def generate_edges(self, new_node):
        edges = []

        for edge in itertools.product([new_node], list(range(new_node))):
            edges.append(list(edge))
            edges.append(list(reversed(edge)))

        return torch.Tensor(edges).int()
    
    def load_viz_wiz(self, num_img, offset=0, fully_connected=True, image_threshold=0.6, word_threshold=0.6):
        vz = VizWiz(offset=offset)
        img_loaded = 0

        while img_loaded is not None and img_loaded < num_img:
            # image, caption, image filename
            i, c, n = vz.__next__()
            
            if c == 'Quality issues are too severe to recognize visual content.':
                continue # skip too noisy images

            print(f'Image caption: {c}')

            self.add_node(i, c, n, fully_connected=fully_connected, image_threshold=image_threshold, word_threshold=word_threshold)
            print(f'Added {img_loaded+1} images.')

            img_loaded += 1

    def get_pyg_graph(self):
        x_img = self.x_img
        x_wrd = self.x_wrd
        edge_index_img = self.edge_index_img.t().contiguous()
        edge_index_wrd = self.edge_index_wrd.t().contiguous()
        edge_index_i2w = self.edge_index_i2w.t().contiguous()
        edge_attr_img = self.edge_attr_img
        edge_attr_wrd = self.edge_attr_wrd
 
        data = HetData(x_img=x_img, x_wrd=x_wrd, edge_index_img=edge_index_img,
                       edge_index_wrd=edge_index_wrd, edge_index_i2w=edge_index_i2w,
                       edge_attr_img=edge_attr_img, edge_attr_wrd=edge_attr_wrd)

        return data

    # TODO: find a way to construct the graph by passing arrays instead of iterating trough them

    def get_stellar_graph(self):
        return StellarGraph.from_networkx(self.g, node_features='feature')

    def save(self, path):
        pkl.dump(self, open(path, 'wb'))

    def load(self, path):
        self = pkl.load(open(path, 'rb'))

class HetData(Data):
    def __init__(self, x_img, x_wrd, edge_index_img, edge_index_wrd, edge_index_i2w, edge_attr_img, edge_attr_wrd, y_img, y_wrd):
        super(HetData, self).__init__()

        self.x_img = x_img
        self.x_wrd = x_wrd
        self.edge_index_img = edge_index_img
        self.edge_index_wrd = edge_index_wrd
        # self.edge_index_i2w = edge_index_i2w
        self.edge_attr_img = edge_attr_img
        self.edge_attr_wrd = edge_attr_wrd

    def __inc__(self, key, value):
        if key == 'edge_index_img':
            return self.x_img.size(0)
        if key == 'edge_index_wrd':
            return self.x_wrd.size(0)
        else:
            return super().__inc__(key, value)
