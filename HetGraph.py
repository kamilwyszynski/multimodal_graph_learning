import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from Img2Vec import Img2Vec
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
import itertools
import os

class HetGraph():

    def __init__(self, english_stopwords_path='res/stopwords/english', word2vec_path='models/word2vec.model'):
        # feature sets for images and words
        self.x_img = torch.Tensor()
        self.x_wrd = torch.Tensor()

        # lists of words and image file names for later tracing
        self.y_img = []
        self.y_wrd = []

        # edges between all types of nodes
        self.edge_index_i2i = torch.Tensor()
        self.edge_index_w2w = torch.Tensor()
        self.edge_index_i2w = torch.Tensor()

        # edge weights
        self.edge_attr_i2i = torch.Tensor()
        self.edge_attr_w2w = torch.Tensor()

        # image feature extraction model
        self.img2vec = Img2Vec()

        # word2vec model
        if os.path.exists(word2vec_path):
            # self.word2vec = gensim.models.Word2Vec.load(word2vec_path)
            self.word2vec = KeyedVectors.load(word2vec_path)
        else:
            self.word2vec = gensim_api.load('glove-wiki-gigaword-300')
            self.word2vec.save(word2vec_path)


        # stopwords in english
        with open(english_stopwords_path, 'r') as file:
            self.stopwords = file.read().split('\n')

    def get_details(self):
        #feature sets for images and words
        print(hg.x_img)
        print(hg.x_wrd)

        # lists of words and image file names for later tracing
        print(hg.y_img)
        print(hg.y_wrd)

        # edges between all types of nodes
        # print(hg.edge_index_i2i)
        # print(hg.edge_index_w2w)
        print(hg.edge_index_i2w)

        # edge weights
        print(hg.edge_attr_i2i)
        print(hg.edge_attr_w2w)        

    def add_node(self, img, cap, img_name):
        # calculate image vector
        img_vec = self.img2vec.get_vector(img)

        # get number of nodes
        num_nodes = self.x_img.size()[0]

        for node_index, saved_img_vec in enumerate(self.x_img):
            # calculate cos similarity between all saved vectors
            img_sim = self.img2vec.get_cos_sim(saved_img_vec, img_vec)
            
            # add similarity as an edge attr
            self.edge_attr_i2i = torch.cat((self.edge_attr_i2i, img_sim.unsqueeze(0), img_sim.unsqueeze(0))) # adding two times because directional

            # add edge between image nodes (both directions)
            edg1 = torch.Tensor([node_index, num_nodes])
            edg2 = torch.Tensor([num_nodes, node_index])
            self.edge_index_i2i = torch.cat((self.edge_index_i2i, edg1.unsqueeze(0), edg2.unsqueeze(0)))

        # add image node and image caption as its label
        self.x_img = torch.cat((self.x_img, img_vec.reshape(1, 512)))
        self.y_img.append([img_name, cap])
        
        # get word list
        words = cap.split()

        for word in words:
            word = word.lower()

            # skip if a stopword
            if (word in self.stopwords) or (word not in self.word2vec.vocab.keys()):
                continue

            if word not in self.y_wrd:
                # print(f'Adding "{word}" to the graph...')

                # add word to the word set
                self.y_wrd.append(word)

                # calculate word vector
                word_vec = self.word2vec.get_vector(word) # numpy

                num_words = self.x_wrd.size()[0] # number of words without the new word

                # if non empty
                if num_words > 0:
                    # get similarities
                    word_sims = self.word2vec.cosine_similarities(word_vec, self.x_wrd.numpy())

                    # add edge weights
                    # repeating each similarity to reflect two-directional edges
                    self.edge_attr_w2w = torch.cat((self.edge_attr_w2w, torch.from_numpy(np.repeat(word_sims, 2))))

                    # add edge indices
                    edge_indices = self.generate_edges(num_words)
                    self.edge_index_w2w = torch.cat((self.edge_index_w2w, edge_indices))

                # add word vector to the node feature set
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
        # feature sets
        x_img = self.x_img
        x_wrd = self.x_wrd

        # edges need transposing
        edge_index_i2i = self.edge_index_i2i.t().contiguous()
        edge_index_w2w = self.edge_index_w2w.t().contiguous()
        edge_index_i2w = self.edge_index_i2w.t().contiguous()

        # edge weights
        edge_attr_i2i = self.edge_attr_i2i
        edge_attr_w2w = self.edge_attr_w2w

        # creating the object
        data = Data(x_img, x_wrd, edge_index_i2i, edge_index_w2w, edge_index_i2w, edge_attr_i2i, edge_attr_w2w)

        return data
