
import sys
import fasttext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc

all_tokens_file = 'ptb_dense_10k.tokens.txt'
all_token_paths_file = 'ptb_dense_10k.token_paths'
corpus = '/home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt'
#corpus = '/home/dave/agi/penn-treebank/simple-examples/data/ptb.train-short.txt'
eos = '<end>'
load = False
show = False
btree = False  # Whether to use a binary-tree encoding
#dense_test_words = ['company', 'increases', 'production']
dense_test_words = []
preprocess = True
preprocessed_corpus = 'postproc.txt'
truncate = False
truncate_size = 100
model_size = 100

base_params = {
  'filename':'model.bin',
  'type':'skipgram',  # or cbow
  'size':model_size
}

model_params = [
  {
    'filename':'skip.model.bin',
    'type':'skipgram'
  },
  {
    'filename':'cbow.model.bin',
    'type':'cbow'
  }
]

def get_param(base_params, model_params, key, model):
  value = base_params[key]
  delta_params = model_params[model]
  if key in delta_params.keys():
    value = delta_params[key]
  print('Model ', m, 'param:', key, 'value:', value)
  return value

# Preprocessing
def preprocess_corpus(input_file, output_file, eos='<end>'):
  """Preprocesses a file to replace line endings with special markers"""
  fin = open(input_file, 'rt')
  fout = open(output_file, 'wt')

  for line in fin:
    line = line.replace('\n', ' '+eos)
    fout.write(line)
    
  fin.close()
  fout.close()

if preprocess:
  preprocess_corpus(corpus, preprocessed_corpus, eos)

# Dense embedding
def test_dense(tokens, token_indices, token_vectors, token):
  num_tokens = len(tokens)
  k = token_indices[token]
  model_size = len(token_vectors[0])
  best_value = sys.float_info.max
  for j in range(0, num_tokens):
    sum_sq_err = 0.0
    for i in range(0, model_size):
      x1 = token_vectors[k,i]
      x2 = token_vectors[j,i]
      xx = x1 * x2
      sq_err = (x1-x2)*(x1-x2)
      sum_sq_err += sq_err
    if sum_sq_err <= best_value:
      best_token = j
      best_value = sum_sq_err
      print('Matching:', token, '(', k, ') Best token:', tokens[best_token], '(', best_token,') Err:', best_value)
  return best_token


# Evaluate models
# https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters
num_models = len(model_params)
models = []
print('Have', num_models, 'models.')
for m in range(0, num_models):
  print('Creating model ', m)
  model_filename = get_param(base_params, model_params, 'filename', m)
  if load:
    model = fasttext.load_model(model_filename)
  else:
    model_type = get_param(base_params, model_params, 'type', m)
    model_size = get_param(base_params, model_params, 'size', m)
    model = fasttext.train_unsupervised(preprocessed_corpus, model=model_type, minCount=1, dim=model_size)
    model.save_model(model_filename)

  num_tokens = len(model.labels)
  print('Model', m, 'has ', num_tokens, 'tokens.')

  # Export from fasttext format
  # Create dense model matrix
  tokens = model.labels
  token_vectors = np.zeros((num_tokens, model_size))
  token_indices = {}
  n = 0
  for label in tokens:
    v = model.get_word_vector(label)
    token_indices[label] = n
    #print(n, ' Label: ', label, ' V: ', v)
    for i in range(0, model_size):
      token_vectors[n,i] = v[i]
    n = n + 1

  num_test_words = len(dense_test_words)
  for t in range(0, num_test_words):
    test_token = dense_test_words[t]
    test_dense(tokens, token_indices, token_vectors, test_token)

  # Option to truncate dataset for visualization
  num_tokens_2 = num_tokens
  token_vectors_2 = token_vectors
  tokens_2 = tokens
  token_indices_2 = token_indices
  if truncate:
    num_tokens_2 = truncate_size
    token_vectors_2 = token_vectors[:num_tokens_2,:]
    tokens_2 = tokens[:num_tokens_2]

  # Store this model
  model_data = {}
  model_data['id'] = m
  model_data['params'] = model_params[m]
  model_data['tokens'] = tokens_2
  model_data['token_indices'] = token_indices_2
  model_data['token_vectors'] = token_vectors_2
  model_data['num_tokens'] = num_tokens_2
  models.append(model_data)

def find_depth(tree, clusters, num_tokens, cluster_id):
  # Calculate the depth of this cluster
  #print('Cluster: ', cluster_id)
  
  linkage_row = tree[cluster_id]
  cluster = clusters[cluster_id]

  depth = cluster['depth']
  if depth > 0:
    #print('Cluster ', cluster_id, ' D=', depth)
    return depth

  child_1 = linkage_row[0]
  child_2 = linkage_row[1]

  depth_1 = 0
  if child_1 < num_tokens:
    #print('Child ', child_1, ' is leaf so D=0')
    pass
  else:
    child_id = int(child_1 - num_tokens)
    depth_1 = find_depth(tree, clusters, num_tokens, child_id)

  depth_2 = 0
  if child_2 < num_tokens:
    #print('Child ', child_2, ' is leaf so D=0')
    pass
  else:
    child_id = int(child_2 - num_tokens)
    depth_2 = find_depth(tree, clusters, num_tokens, child_id)

  depth = max(depth_1, depth_2) +1

  #print('Cluster ', cluster_id, ' D=', depth)
  cluster['depth'] = depth
  return depth

def eval_node(num_tokens, token_id, node_id, path, fork):
  path2 = path.copy()
  path2.append(fork)
  if node_id < num_tokens: # is leaf
    if node_id == token_id:
      #print('FOUND: Node ', node_id, ' is leaf')
      return path2, None  # Correct match
    return None, None
  else:  # Not leaf
    child_cluster_id = int(node_id - num_tokens)
    #print('Non leaf, node ', node_id, ' as cluster ', child_cluster_id)
    return path2, child_cluster_id

def find_path(tree, num_tokens, token_id, cluster_id, path):
  # Find the path in the tree to token_id
  #print('Find path for cluster: ', cluster_id, ' path: ', path)
  
  linkage_row = tree[cluster_id]
  node_0 = linkage_row[0]
  node_1 = linkage_row[1]

  path_0, child_cluster_id_0 = eval_node(num_tokens, token_id, node_0, path, fork=0)
  path_1, child_cluster_id_1 = eval_node(num_tokens, token_id, node_1, path, fork=1)

  path_0b = None
  if child_cluster_id_0 is not None:
    #print('Recurse 0')
    path_0b = find_path(tree, num_tokens, token_id, child_cluster_id_0, path_0)
  elif path_0 is not None:
    #print('Found 0')
    return path_0  # solution

  path_1b = None
  if child_cluster_id_1 is not None:
    #print('Recurse 1')
    path_1b = find_path(tree, num_tokens, token_id, child_cluster_id_1, path_1)
  elif path_1 is not None:
    #print('Found 1')
    return path_1  # solution

  if path_0b is not None:
    #print('R-Found 0')
    return path_0b
  if path_1b is not None:
    #print('R-Found 1')
    return path_1b
  #print('Not found.')
  return None

vector_size = model_size
vector_key = 'token_vectors'

if btree:
  sys.setrecursionlimit(10000)
  max_tree_depth = 0

  for m in range(0, num_models):
    print('Clustering model ', m)
    model_data = models[m]
    tokens = model_data['tokens']
    token_vectors = model_data['token_vectors']
    num_tokens = model_data['num_tokens']

    # Cluster the token vectors
    tree = shc.linkage(token_vectors, method='ward')

    if show:
      plt.figure(figsize=(10, 7))
      plt.title('Binary word-tree')
      dend = shc.dendrogram(tree, labels=tokens)
      plt.show()

    #https://stackoverflow.com/questions/9838861/scipy-linkage-format
    #print('Linkage: ', lnk)

    # Next step - convert linkage to tree-paths matrix
    #"A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices
    # Z[i, 0] and Z[i, 1] are combined to form cluster n + i. A cluster with an index 
    # less than n corresponds to one of the original observations. The distance between 
    # clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents
    # the number of original observations in the newly formed cluster."
    #     ?            ?             Dist         Obs in cluster
    #  [[ 10.          29.           0.90266106   2.        ]
    #  [  8.          59.           0.96037416   2.        ]
    #  [  0.           4.           1.10519679   2.        ]
    #  [  1.           2.           1.18796531   2.        ]
    #  [ 12.          21.           1.21003461   2.        ]
    #  [100.         103.           1.29104273   4.        ]
    #  [ 63.          66.           1.2961218    2.        ]
    #  [ 13.          93.           1.33565727   2.        ]
    #  [ 23.          28.           1.33757345   2.        ]
    # ...
    # [183.         185.           3.86640589  15.        ]
    # [162.         180.           4.00119524   6.        ]
    # [161.         189.           4.01574254  33.        ]
    # [188.         190.           4.19644353  46.        ]
    # [179.         192.           4.39205466  39.        ]
    # [187.         191.           5.05303151  11.        ]
    # [193.         194.           5.43354232  85.        ]
    # [178.         195.           5.48807551  15.        ]
    # [196.         197.           7.74025727 100.        ]]

    linkage_shape = tree.shape
    #print('linkage shape: ', linkage_shape)
    linkage_rows = linkage_shape[0]

    # Init clusters
    num_clusters = linkage_rows
    clusters = []  # cluster info
    for j in range(0, num_clusters):
      clusters.append({})
      clusters[j]['depth'] = 0

    # Calc tree depth
    max_depth = 0
    for j in range(0, num_clusters):
      #print('Calculate depth of cluster: ', j)
      depth = find_depth(tree, clusters, num_tokens, j)
      max_depth = max(depth, max_depth)
      #print('Depth of cluster: ', j, ' is ', depth)

    # Build the decision tree paths for each token
    token_paths = np.zeros([num_tokens, max_depth])

    #for i in range(0, 3):
    for i in range(0, num_tokens):
      token = tokens[i]
      j = num_clusters -1
      path = find_path(tree, num_tokens, i, j, [])
      #print('Path of word: ', i, ' which is ', token, ' is ', path)
      path_length = len(path)
      for k in range(0, path_length):
        token_paths[i][k] = path[k]

    print('Max depth = ', max_depth)
    model_data['tree_depth'] = max_depth
    model_data['token_paths'] = token_paths

    max_tree_depth = max(max_tree_depth, max_depth)

  model_size = max_tree_depth  # discovered above
  vector_key = 'token_paths'

all_tokens = None
all_token_paths = np.zeros([num_tokens, num_models, model_size])

for m in range(0, num_models):
  print('Combining tree ', m)
  model_data = models[m]
  tokens = model_data['tokens']
  token_indices = model_data['token_indices']
  token_vectors = model_data[vector_key]

  if m == 0:
    all_tokens = tokens  # Copy from 1st model

  for t in range(0, num_tokens):
    token = all_tokens[t]  # Always use same token in
    index = token_indices[token]
    vector = token_vectors[index]
    vector_length = len(vector)

    for i in range(0, vector_length):
      all_token_paths[t][m][i] = vector[i]
    for i in range(vector_length, model_size):
      all_token_paths[t][m][i] = 2

# https://stackoverflow.com/questions/48230230/typeerror-mismatch-between-array-dtype-object-and-format-specifier-18e?rq=1
np.set_printoptions(threshold=np.nan)
#print('Token paths: \n', all_token_paths)
delimiter = ','
np.savetxt(all_tokens_file, all_tokens, delimiter=delimiter, fmt='%s')
np.save(all_token_paths_file, all_token_paths)
# np.savetxt(all_token_paths_file, all_token_paths, delimiter=delimiter, fmt='%s')