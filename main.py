# from Img2Vec import Img2Vec
# import os
# import json
# import VizWiz
# import PIL
import torch
import itertools
import pickle
from HetGraph import HetGraph
from VizWiz import VizWiz

# create the hg object
hg = HetGraph()

# load an image
vz = VizWiz()

number_of_images = 5
images_loaded = 0

while images_loaded is not None and images_loaded < number_of_images:
    # get next image and caption
    i, c, n = vz.__next__()

    # skip too noisy images
    if c == 'Quality issues are too severe to recognize visual content.':
        continue

    print(f'Image caption: {c}')

    hg.add_node(i, c, n)
    print(f'Added {images_loaded+1} images.')

    images_loaded += 1

pickle.dump(hg, open(f'res/graphs/{number_of_images}_img_non_pyg.hg', 'wb'))

# feature sets for images and words
print(hg.x_img)
print(hg.x_wrd)

# lists of words and image file names for later tracing
print(hg.y_img)
print(hg.y_wrd)

# edges between all types of nodes
print(hg.edge_index_i2i)
print(hg.edge_index_w2w)
print(hg.edge_index_i2w)

# edge weights

print(hg.edge_attr_i2i)
print(hg.edge_attr_w2w)



# def generate_edges(new_node):
#     edges = []

#     for edge in itertools.product([new_node], list(range(new_node))):
#         edges.append(list(edge))
#         edges.append(list(reversed(edge)))

#     return torch.Tensor(edges).int()

# print(generate_edges(2))