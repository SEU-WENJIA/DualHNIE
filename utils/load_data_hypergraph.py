import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import pickle
import numpy as np
from collections import defaultdict
from itertools import combinations



def get_hyperedge_types(edges, edge_type, hyperedges):
    """
    Generate hyperedge types based on the edge types of node pairs within each hyperedge.

    :param edges: A tuple of two tensors, representing the source and destination nodes of edges.
    :param edge_type: A tensor representing the types of each edge.
    :param hyperedges: A list of lists, where each sublist contains the nodes in a hyperedge.
    :return: A tensor where each element represents the type of a hyperedge.
    """
    # Create a dictionary to map node pairs to their edge type
    node_pair_to_type = {}
    for i in range(len(edges[0])):
        u = edges[0][i].item()
        v = edges[1][i].item()
        rel = edge_type[i].item()
        # Use a frozenset to represent the node pair for undirected edges
        key = frozenset({u, v})
        node_pair_to_type[key] = rel

    # Create a list to store the types of each hyperedge
    hyperedge_types = []
    for hyperedge in hyperedges:
        if len(hyperedge) < 2:
            # Assign a default type for hyperedges with a single node
            hyperedge_types.append(tuple([-1]))   
        else:
            # Generate all possible node pairs within the hyperedge
            node_pairs = combinations(hyperedge, 2)
            # Collect the edge types for each node pair
            rels = []
            for pair in node_pairs:
                node_pair = frozenset(pair)
                if node_pair in node_pair_to_type:
                    rels.append(node_pair_to_type[node_pair])
            # Remove duplicate relation types and sort them for consistency
            unique_rels = sorted(list(set(rels)))
            hyperedge_types.append(tuple(unique_rels))

    # Assign a unique identifier to each unique combination of relation types
    unique_type_combinations = list(set(hyperedge_types))
    type_id_map = {rel_tuple: idx for idx, rel_tuple in enumerate(unique_type_combinations)}

    # Convert the hyperedge types to a tensor of unique identifiers
    hyperedge_type_tensor = torch.tensor([type_id_map[tuple(rels)] for rels in hyperedge_types], dtype=torch.long)

    return hyperedge_type_tensor



def build_hypergraph(rel_to_entities, num_entities, edges, edge_types):
    """
    Build hypergraph from relation-to-entity mapping.

    Args:
        rel_to_entities (dict): Mapping from relation type to list of entities.
        num_entities (int): Total number of entities/nodes.
        edges (list or np.ndarray): Original edges in the graph.
        edge_types (list or np.ndarray): Types of edges aligned with `edges`.

    Returns:
        hyperedges (list[list]): List of hyperedges, each being a list of nodes.
        node_hyperedge_array (list[tuple]): List of (node, hyperedge_id) pairs.
        hyperedge_types (list): Type information for each hyperedge.
    """
    # Step 1: Map each entity to its set of relations
    entity_to_relations = {node: set() for node in range(num_entities)}
    for rel_type, entities in rel_to_entities.items():
        for entity in entities:
            if entity < num_entities:
                entity_to_relations[entity].add(rel_type)

    # Step 2: Map each unique relation set to nodes sharing that set
    relation_to_nodes = defaultdict(list)
    for node, rels in entity_to_relations.items():
        rel_set = frozenset(rels)  # frozenset to use as dict key
        relation_to_nodes[rel_set].append(node)

    # Step 3: Create hyperedges
    hyperedges = []
    hyperedge_id_map = {}
    hyperedge_id = 0
    for rel_set, nodes in relation_to_nodes.items():
        if len(nodes) > 0:
            hyperedges.append(nodes)
            for node in nodes:
                hyperedge_id_map[(node, rel_set)] = hyperedge_id
            hyperedge_id += 1

    # Step 4: Create node-hyperedge pairs
    node_hyperedge_array = [
        (node, hyperedge_id_map[(node, frozenset(entity_to_relations[node]))])
        for node in entity_to_relations
        if (node, frozenset(entity_to_relations[node])) in hyperedge_id_map
    ]

    # Step 5: Compute hyperedge types
    hyperedge_types = get_hyperedge_types(edges, edge_types, hyperedges)

    return hyperedges, node_hyperedge_array, hyperedge_types





def load_fb15k_rel_data(data_path, cross_validation_shift=0, dataset_name='FB15k_rel'):
    """
    Load FB15k data and construct a hypergraph with shared relations.
    
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split for cross-validation
    :return: A tuple containing the hypergraph, edge types, edge norm, relation number, node features, labels, and indices

    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']   # [2, num_edges]
    labels = data['labels']  # [num_nodes]

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    else:
        node_feats = data['features']  # [num_nodes, num_feature]
    
    # edge list
    edges = data['edges']   # [2, num_edges]
    labels = data['labels']  # [num_nodes]    
    invalid_masks = data['invalid_masks']  # [num_nodes, 1]
    edge_types = data['edge_types']   # [num_edges,1] 
    rel_num = (max(edge_types) + 1).item()
    # rel_num = 30

    # Create a dictionary to hold the entities involved in each relation type
    # [item1 relation item2]  ——>   relation i: [item i_1, item i_2,  ， item i_k] 
    rel_to_entities = {rel: set() for rel in range(rel_num)}

    for i, rel_type in enumerate(edge_types):
        rel_to_entities[rel_type.item()].add(edges[0][i].item())  # Add head entity
        rel_to_entities[rel_type.item()].add(edges[1][i].item())  # Add tail entity



    # Create a dictionary to hold the relations that each entity participates in
    all_entities = set(edges[0].tolist() + edges[1].tolist())
    num_entities = max(all_entities) + 1 


    hyperedges, node_hyperedge_array, hyperedge_types = build_hypergraph(rel_to_entities, num_entities, edges, edge_types)


    dataset_dir = './datasets/fb15k'
    hyperedge_types_path = os.path.join(dataset_dir, 'hyperedge_types.pt')
    hyperedges_path = os.path.join(dataset_dir, 'hyperedges.npy')
    node_hyperedge_array_path = os.path.join(dataset_dir, 'node_hyperedge_array.npy')


    if os.path.exists(hyperedge_types_path) and os.path.exists(hyperedges_path) and os.path.exists(node_hyperedge_array_path):

        hyperedge_types = torch.load(hyperedge_types_path)
        hyperedges = np.load(hyperedges_path, allow_pickle=True)
        node_hyperedge_array = np.load(node_hyperedge_array_path, allow_pickle=True).tolist()
    else:
        hyperedges, node_hyperedge_array, hyperedge_types = build_hypergraph(
            rel_to_entities, num_entities, edges, edge_types
        )

        
        os.makedirs(dataset_dir, exist_ok=True)
        torch.save(hyperedge_types, hyperedge_types_path)
        np.save(hyperedges_path, hyperedges)
        np.save(node_hyperedge_array_path, node_hyperedge_array, allow_pickle=True)


    rel_num = (max(hyperedge_types)+1).item()
    hg = dgl.heterograph({('node', 'to_hyperedge', 'hyperedge'): node_hyperedge_array})

    H = np.zeros((num_entities, len(hyperedges)))
    entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}  
    for hyperedge_id, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            if node in entity_to_index:
                H[entity_to_index[node], hyperedge_id] = 1
    H = torch.tensor(H, dtype=torch.float32)


    # generate edge norm
    in_deg = H.sum(-1)
    norm = 1.0 / in_deg
    norm[torch.isinf(norm)] = 0
    node_norm = norm.view(-1,1)
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(num_entities) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]

    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
        
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

 
    if 'two' in dataset_name:
        return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
        
    return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feats, labels, train_idx, val_idx, test_idx





def load_imdb_s_rel_data(data_path, cross_validation_shift=0, dataset_name='IMDB_S_rel'):
    """
    Load FB15k data and construct a hypergraph with shared relations.    
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split for cross-validation
    :return: A tuple containing the hypergraph, edge types, edge norm, relation number, node features, labels, and indices
    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/imdb_s_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'two' in dataset_name:

        node_feat1 = pickle.load(open('./datasets/imdb_s_node2vec.pk', 'rb'))
        node_feat2 = pickle.load(open('./datasets/imdb_s_lang.pk', 'rb'))
        
    elif 'concat' in dataset_name:
        node_feat1 = torch.from_numpy(pickle.load(open('./datasets/imdb_s_node2vec.pk', 'rb')))
        node_feat2 = pickle.load(open('./datasets/imdb_s_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], 1)
    else:
        node_feats = torch.from_numpy(pickle.load(open('./datasets/imdb_s_node2vec.pk', 'rb')))

    
    # edge list
    edges = data['edges']   # [2, num_edges]
    labels = data['labels']  # [num_nodes]    
    invalid_masks = data['invalid_masks']  # [num_nodes, 1]
    edge_types = data['edge_types']   # [num_edges,1] 
    rel_num = (max(edge_types) + 1).item()
    # rel_num = 30

    # Create a dictionary to hold the entities involved in each relation type
    # [item1 relation item2]  ——>   relation i: [item i_1, item i_2,  ， item i_k] 
    rel_to_entities = {rel: set() for rel in range(rel_num)}
    for i, rel_type in enumerate(edge_types):
        rel_to_entities[rel_type.item()].add(edges[0][i].item())  # Add head entity
        rel_to_entities[rel_type.item()].add(edges[1][i].item())  # Add tail entity

    # Create a dictionary to hold the relations that each entity participates in
    all_entities = set(edges[0].tolist() + edges[1].tolist())
    num_entities = max(all_entities) + 1 


    dataset_dir = './datasets/imdb/'
    hyperedge_types_path = os.path.join(dataset_dir, 'hyperedge_types.pt')
    hyperedges_path = os.path.join(dataset_dir, 'hyperedges.npy')
    node_hyperedge_array_path = os.path.join(dataset_dir, 'node_hyperedge_array.npy')


    if os.path.exists(hyperedge_types_path) and os.path.exists(hyperedges_path) and os.path.exists(node_hyperedge_array_path):
       
        hyperedge_types = torch.load(hyperedge_types_path)
        hyperedges = np.load(hyperedges_path, allow_pickle=True)
        node_hyperedge_array = np.load(node_hyperedge_array_path, allow_pickle=True).tolist()
    else:
        # if not, generate data
        hyperedges, node_hyperedge_array, hyperedge_types = build_hypergraph(
            rel_to_entities, num_entities, edges, edge_types
        )

        
        os.makedirs(dataset_dir, exist_ok=True)
        torch.save(hyperedge_types, hyperedge_types_path)
        np.save(hyperedges_path, hyperedges)
        np.save(node_hyperedge_array_path, node_hyperedge_array, allow_pickle=True)


    rel_num = (max(hyperedge_types)+1).item()
    hg = dgl.heterograph({('node', 'to_hyperedge', 'hyperedge'): node_hyperedge_array})

    H = np.zeros((num_entities, len(hyperedges)))
    entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}  
    for hyperedge_id, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            if node in entity_to_index:
                H[entity_to_index[node], hyperedge_id] = 1
    H = torch.tensor(H, dtype=torch.float32)

    

    # generate edge norm
    in_deg = H.sum(-1)
    norm = 1.0 / in_deg
    norm[torch.isinf(norm)] = 0
    node_norm = norm.view(-1,1)
    labels = torch.log(1 + labels)


 
    # split dataset
    float_mask = np.ones(num_entities) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

 
    if 'two' in dataset_name:
        return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
        
    return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feats, labels, train_idx, val_idx, test_idx



def load_tmdb_rel_data(data_path, cross_validation_shift=0, dataset_name='TMDB_rel'):
    """
    Load FB15k data and construct a hypergraph with shared relations.
    
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split for cross-validation
    :return: A tuple containing the hypergraph, edge types, edge norm, relation number, node features, labels, and indices

    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)


    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    else:
        node_feats = data['features']  # [num_nodes, num_feature]
    

    # edge list
    edges = data['edges']   # [2, num_edges]
    labels = data['labels']  # [num_nodes]    
    invalid_masks = data['invalid_masks']  # [num_nodes, 1]
    edge_types = data['edge_types']   # [num_edges,1] 
    rel_num = (max(edge_types) + 1).item()
    # rel_num = 30


    print(edges[0].shape)

    print(labels.shape)
    # Create a dictionary to hold the entities involved in each relation type
    # [item1 relation item2]  ——>   relation i: [item i_1, item i_2,  ， item i_k] 
    rel_to_entities = {rel: set() for rel in range(rel_num)}
    for i, rel_type in enumerate(edge_types):
        rel_to_entities[rel_type.item()].add(edges[0][i].item())  # Add head entity
        rel_to_entities[rel_type.item()].add(edges[1][i].item())  # Add tail entity

    # Create a dictionary to hold the relations that each entity participates in
    all_entities = set(edges[0].tolist() + edges[1].tolist())
    num_entities = max(all_entities) + 1 


    entity_to_relations = {node: set() for node in range(num_entities)}
    for rel_type, entities in rel_to_entities.items():
        for entity in entities:
            if entity < num_entities:
                entity_to_relations[entity].add(rel_type)

    dataset_dir = './datasets/tmdb5k'
    hyperedge_types_path = os.path.join(dataset_dir, 'hyperedge_types.pt')
    hyperedges_path = os.path.join(dataset_dir, 'hyperedges.npy')
    node_hyperedge_array_path = os.path.join(dataset_dir, 'node_hyperedge_array.npy')


    if os.path.exists(hyperedge_types_path) and os.path.exists(hyperedges_path) and os.path.exists(node_hyperedge_array_path):

        hyperedge_types = torch.load(hyperedge_types_path)
        hyperedges = np.load(hyperedges_path, allow_pickle=True)
        node_hyperedge_array = np.load(node_hyperedge_array_path, allow_pickle=True).tolist()
    else:
        # if not, generate data
        hyperedges, node_hyperedge_array, hyperedge_types = build_hypergraph(
            rel_to_entities, num_entities, edges, edge_types
        )

        
        os.makedirs(dataset_dir, exist_ok=True)
        torch.save(hyperedge_types, hyperedge_types_path)
        np.save(hyperedges_path, hyperedges)
        np.save(node_hyperedge_array_path, node_hyperedge_array, allow_pickle=True)


    rel_num = (max(hyperedge_types)+1).item()
    hg = dgl.heterograph({('node', 'to_hyperedge', 'hyperedge'): node_hyperedge_array})

    H = np.zeros((num_entities, len(hyperedges)))
    entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}  
    for hyperedge_id, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            if node in entity_to_index:
                H[entity_to_index[node], hyperedge_id] = 1
    H = torch.tensor(H, dtype=torch.float32)

    

    # generate edge norm
    in_deg = H.sum(-1)
    norm = 1.0 / in_deg
    norm[torch.isinf(norm)] = 0
    node_norm = norm.view(-1,1)
    labels = torch.log(1 + labels)


 
    # split dataset
    float_mask = np.ones(num_entities) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

 
    if 'two' in dataset_name:
        return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
        
    return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feats, labels, train_idx, val_idx, test_idx




def load_music10k_rel_data(data_path, cross_validation_shift=0, dataset_name='music10k_rel'):
    """
    Load music10k data and construct a hypergraph with shared relations.
    
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split for cross-validation
    :return: A tuple containing the hypergraph, edge types, edge norm, relation number, node features, labels, and indices

    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']   # [2, num_edges]
    labels = data['labels']  # [num_nodes]

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/music_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/music_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/music_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    else:
        node_feats = data['features']  # [num_nodes, num_feature]
    
    # edge list
    edges = data['edges']   # [2, num_edges]
    labels = data['labels']  # [num_nodes]    
    invalid_masks = data['invalid_masks']  # [num_nodes, 1]
    edge_types = data['edge_types']   # [num_edges,1] 
    rel_num = (max(edge_types) + 1).item()
    # rel_num = 30

    # Create a dictionary to hold the entities involved in each relation type
    # [item1 relation item2]  ——>   relation i: [item i_1, item i_2,  ， item i_k] 
    rel_to_entities = {rel: set() for rel in range(rel_num)}

    for i, rel_type in enumerate(edge_types):
        rel_to_entities[rel_type.item()].add(edges[0][i].item())  # Add head entity
        rel_to_entities[rel_type.item()].add(edges[1][i].item())  # Add tail entity



    # Create a dictionary to hold the relations that each entity participates in
    all_entities = set(edges[0].tolist() + edges[1].tolist())
    num_entities = max(all_entities) + 1 


    entity_to_relations = {node: set() for node in range(num_entities)}
    for rel_type, entities in rel_to_entities.items():
        for entity in entities:
            if entity < num_entities:
                entity_to_relations[entity].add(rel_type)

    dataset_dir = './datasets/music10k'
    hyperedge_types_path = os.path.join(dataset_dir, 'hyperedge_types.pt')
    hyperedges_path = os.path.join(dataset_dir, 'hyperedges.npy')
    node_hyperedge_array_path = os.path.join(dataset_dir, 'node_hyperedge_array.npy')


    if os.path.exists(hyperedge_types_path) and os.path.exists(hyperedges_path) and os.path.exists(node_hyperedge_array_path):

        hyperedge_types = torch.load(hyperedge_types_path)
        hyperedges = np.load(hyperedges_path, allow_pickle=True)
        node_hyperedge_array = np.load(node_hyperedge_array_path, allow_pickle=True).tolist()
    else:
        # if not, generate data
        hyperedges, node_hyperedge_array, hyperedge_types = build_hypergraph(
            rel_to_entities, num_entities, edges, edge_types
        )

        
        os.makedirs(dataset_dir, exist_ok=True)
        torch.save(hyperedge_types, hyperedge_types_path)
        np.save(hyperedges_path, hyperedges)
        np.save(node_hyperedge_array_path, node_hyperedge_array, allow_pickle=True)



    rel_num = (max(hyperedge_types)+1).item()
    hg = dgl.heterograph({('node', 'to_hyperedge', 'hyperedge'): node_hyperedge_array})

    H = np.zeros((num_entities, len(hyperedges)))
    entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}  
    for hyperedge_id, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            if node in entity_to_index:
                H[entity_to_index[node], hyperedge_id] = 1

    H = torch.tensor(H, dtype=torch.float32)

    # generate edge norm
    in_deg = H.sum(-1)
    norm = 1.0 / in_deg
    norm[torch.isinf(norm)] = 0
    node_norm = norm.view(-1,1)
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(num_entities) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

 
    if 'two' in dataset_name:
        return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
        
    return hg, hyperedge_types, H ,hyperedges, edges,   rel_num, node_feats, node_feats, labels, train_idx, val_idx, test_idx



def load_data(data_path, cross_validation_shift, dataset_name):

    if 'FB15k' in dataset_name:
        data_path = './datasets/fb15k_rel.pk'
        g,  edge_types, H, hyperedges, edges,  rel_num, struct_feats, semantic_feats, labels, train_idx, val_idx, test_idx =  \
            load_fb15k_rel_data(data_path, cross_validation_shift=0, dataset_name='FB15k_rel_two')

    elif 'IMDB_S' in dataset_name:
        data_path = './datasets/imdb_s_rel.pk'
        g,  edge_types, H, hyperedges, edges,  rel_num, struct_feats, semantic_feats, labels, train_idx, val_idx, test_idx =  \
            load_imdb_s_rel_data(data_path, cross_validation_shift=0, dataset_name='IMDB_S_rel_two')

    elif 'TMDB' in dataset_name:

        data_path = './datasets/tmdb_rel.pk'
        g,  edge_types, H, hyperedges, edges,  rel_num, struct_feats, semantic_feats, labels, train_idx, val_idx, test_idx =  \
            load_tmdb_rel_data(data_path, cross_validation_shift=0, dataset_name='TMDB_rel_two')

    elif 'MUSIC'  in dataset_name or 'music' in dataset_name:
        data_path  = './datasets/music_rel.pk'
        g,  edge_types, H, hyperedges, edges,  rel_num, struct_feats, semantic_feats, labels, train_idx, val_idx, test_idx =  \
            load_music10k_rel_data(data_path, cross_validation_shift=0, dataset_name='MUSIC_rel')

    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}!')
    

    return g,  edge_types, H, hyperedges, edges,  rel_num, struct_feats, semantic_feats, labels, train_idx, val_idx, test_idx




