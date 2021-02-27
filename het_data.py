import torch
from torch_geometric.data import Data
import numpy as np

from img2vec import Img2Vec
from dataset import VizWiz

import gensim.downloader as gensim_api
from gensim.models import KeyedVectors

import itertools
import os
import pickle as pkl

from torch_geometric.data import DataLoader

import networkx as nx
from stellargraph import StellarGraph

class HetGraph():

    def __init__(self, english_stopwords_path='res/stopwords/english', word2vec_path='models/word2vec.model'):
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

    # TODO: cleanup the process
    def add_node(self, img, cap, img_name):
        img_vec = self.img2vec.get_vector(img) # calculate image vector
        num_nodes = self.x_img.size()[0] # get number of nodes

        for node_index, saved_img_vec in enumerate(self.x_img):
            img_sim = self.img2vec.get_cos_sim(saved_img_vec, img_vec) # calculate cos similarity between all saved vectors
            self.edge_attr_img = torch.cat((self.edge_attr_img, img_sim.unsqueeze(0), img_sim.unsqueeze(0))) # adding two times because directional

            # add edge between image nodes (both directions)
            edg1 = torch.Tensor([node_index, num_nodes])
            edg2 = torch.Tensor([num_nodes, node_index])
            self.edge_index_img = torch.cat((self.edge_index_img, edg1.unsqueeze(0), edg2.unsqueeze(0)))

        # add image node and image caption as its label
        self.x_img = torch.cat((self.x_img, img_vec.reshape(1, 512)))
        self.y_img.append([img_name, cap])

        for word in cap.split():
            word = word.lower()

            if (word in self.stopwords) or (word not in self.word2vec.vocab.keys()):
                continue # skip if a stopword

            if word not in self.y_wrd:
                self.y_wrd.append(word)

                word_vec = self.word2vec.get_vector(word) # calculate word vector

                num_words = self.x_wrd.size()[0] # number of words without the new word

                if num_words > 0:
                    word_sims = self.word2vec.cosine_similarities(word_vec, self.x_wrd.numpy())

                    # repeating each similarity to reflect two-directional edges
                    self.edge_attr_wrd = torch.cat((self.edge_attr_wrd, torch.from_numpy(np.repeat(word_sims, 2))))

                    edge_indices = self.generate_edges(num_words)
                    self.edge_index_wrd = torch.cat((self.edge_index_wrd, edge_indices))

                # need to reshape before cat for later usage
                self.x_wrd = torch.cat((self.x_wrd, torch.from_numpy(word_vec.reshape(1, 300))))

                # add new word to image edges (not directional)
                img_idx = num_nodes
                wrd_idx = num_words
                edg = torch.Tensor([img_idx, wrd_idx])
                self.edge_index_i2w = torch.cat((self.edge_index_i2w, edg.unsqueeze(0))).int()
            
            elif word in self.y_wrd:

                # add existing word to image edges (not directional)
                img_idx = num_nodes
                wrd_idx = self.y_wrd.index(word)
                edg = torch.Tensor([img_idx, wrd_idx])
                self.edge_index_i2w = torch.cat((self.edge_index_i2w, edg.unsqueeze(0))).int()

    def generate_edges(self, new_node):
        edges = []

        for edge in itertools.product([new_node], list(range(new_node))):
            edges.append(list(edge))
            edges.append(list(reversed(edge)))

        return torch.Tensor(edges).int()

    def get_data_object(self):
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
    
    def load_viz_wiz(self, num_img):
        vz = VizWiz()
        img_loaded = 0

        while img_loaded is not None and img_loaded < num_img:
            # image, caption, image filename
            i, c, n = vz.__next__()
            
            if c == 'Quality issues are too severe to recognize visual content.':
                continue # skip too noisy images

            print(f'Image caption: {c}')

            self.add_node(i, c, n)
            print(f'Added {img_loaded+1} images.')

            img_loaded += 1

    def get_stellar_graph(self):
        g = nx.Graph()

        for edge, sim in zip(self.edge_index_img[1::2], self.edge_attr_img[1::2]):
            node1 = self.y_img[int(edge[0])][0]
            node2 = self.y_img[int(edge[1])][0]

            feature1 = self.x_img[int(edge[0])].numpy()
            feature2 = self.x_img[int(edge[1])].numpy()

            g.add_edge(node1, node2, weight=float(sim), label='image2image')

            g.nodes[node1]['label'] = 'image'
            g.nodes[node2]['label'] = 'image'

            g.nodes[node1]['feature'] = feature1
            g.nodes[node2]['feature'] = feature2


        for edge, sim in zip(self.edge_index_wrd[1::2], self.edge_attr_wrd[1::2]):
            node1 = self.y_wrd[int(edge[0])][0]
            node2 = self.y_wrd[int(edge[1])][0]

            feature1 = self.x_wrd[int(edge[0])].numpy()
            feature2 = self.x_wrd[int(edge[1])].numpy()

            g.add_edge(node1, node2, weight=float(sim), label='word2word')

            g.nodes[node1]['label'] = 'word'
            g.nodes[node2]['label'] = 'word'

            g.nodes[node1]['feature'] = feature1
            g.nodes[node2]['feature'] = feature2

        
        for edge in self.edge_index_i2w:
            node1 = self.y_img[int(edge[0])][0]
            node2 = self.y_wrd[int(edge[1])][0]

            g.add_edge(node1, node2, label='image2word')
        
        return StellarGraph.from_networkx(g, node_features='feature')

    def save(self, path):
        pkl.dump(self, open(path, 'wb'))

    def load(path):
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
