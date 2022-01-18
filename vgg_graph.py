import pandas as pd
import json
import community as community_louvain
import networkx as nx

# sample graph
# TODO: take graph as parameter no need for edges_df...
def subgraph_sample(partition_df, count, graph):
    view_partition = partition_df.copy()
    # - sort by degree, 'u' is kept, so high degree first
    sub_graph = graph.copy()
    if count > len(sub_graph.nodes()):
        return sub_graph

    sub_degree = pd.DataFrame(nx.degree(sub_graph), columns=['node', 'degree'])
    view_partition = view_partition.merge(sub_degree, on='node')
    for i in range(9):
        # should be apply above..but requires sub_graph parallel...so each part graph handled separately??
        view_partition = view_partition[view_partition['node'].isin(sub_graph.nodes)]
        for part, grp in view_partition.sort_values('degree', ascending=False).groupby('part'):
            grp['odd'] = grp.reset_index(drop=True).index // 2
            grp = grp[:grp.shape[0]-(grp.shape[0] % 2)] # select frac & trim odd counts need pairs 
            # grp = grp[::2] # select frac & trim odd counts need pairs 
            for odd, pair in grp.groupby('odd'):  # trim odd counts need pairs
                nx.contracted_nodes(sub_graph, pair.node.values[0], pair.node.values[1], copy=False, self_loops=False)
        print(len(sub_graph.nodes), 'nodes')
        if count > len(sub_graph.nodes()):
            break

    print('shape:', len(sub_graph.nodes), 'nodes')
    return sub_graph


def create_nodes_df(partition_df, sub_graph, manual_y_df):
    # create nodes_df
    nodes_df = pd.DataFrame(sub_graph.nodes()).merge(manual_y_df, left_on=0, right_on='im_id',how='left') # !!!
    nodes_df = nodes_df.merge(partition_df, left_on=0, right_on='node',how='left')

    part_count = pd.concat([partition_df['part'].value_counts(), nodes_df['part'].value_counts().rename('sub')], axis=1).fillna(0)
    part_count = (part_count['part'] - part_count['sub']).rename('part_diff').reset_index()
    nodes_df = nodes_df.merge(part_count, left_on='part', right_on='index')

    # nodes have things behind them...
    nodes_df['shape'] = 'tri'
    nodes_df.loc[nodes_df['part_diff'] < 10, 'shape'] = 'rect'
    nodes_df.loc[nodes_df['part_diff'] == 0, 'shape'] = 'circle'

    nodes_df['y'] = nodes_df['y'].fillna(4)
    nodes_df = nodes_df.rename(columns={'y':'group', 0:'id'})
    nodes_df = nodes_df.fillna('None')
    nodes_df['img_name'] = '/img/' + nodes_df['img_name']
    print('nodes_df created !', nodes_df.shape[0])
    if 'level' in nodes_df.columns:
        nodes_dict = nodes_df[['id', 'group', 'part', 'shape', 'level']].to_dict(orient='index')
    else:
        nodes_dict = nodes_df[['id', 'group', 'part', 'shape']].to_dict(orient='index')
    nodes_list = list(nodes_dict.values())
    return nodes_df, nodes_list

# create links_df
def create_links_df(sub_graph):
    links_df = pd.DataFrame([sub_graph.edges()]).T
    links_df = links_df.reset_index().drop(0, axis=1)
    links_df = pd.concat([links_df, links_df['index'].apply(lambda x:pd.Series(x))], axis=1).drop(['index'], axis=1)
    print('links_df created !', links_df.shape[0])
    edges_dict = links_df.rename(columns={0: 'source', 1: 'target', 'weight': 'value'}).to_dict(orient='index')
    edges_list = list(edges_dict.values())
    return links_df, edges_list

def create_jsons(nodes_list, nodes_df, edges_list, data_name, part_num, thr):
    # create json file
    store_json = {'nodes':nodes_list, 'links':edges_list}
    with open("./js/" + data_name + "_thr_" + str(thr) + '_p' + str(part_num) + ".json" , "w") as outfile:
        json.dump(store_json, outfile)

    # create *image* json file
    store_json = nodes_df[['img_name','id']].set_index('id')['img_name'].to_dict()
    with open("./js/" + data_name + "_thr_" + str(thr) + '_p' + str(part_num) + "_img.json" , "w") as outfile:
        json.dump(store_json, outfile)

    print('part', part_num, 'is ready!')
    print('path:',data_name + "_thr_" + str(thr) + '_p' + str(part_num))

def get_data():
    prediction_df = pd.read_parquet('prediction_df.' + model_name + '.parq')
    # prediction_df['true_class'] = prediction_df['img_path'].apply(lambda x: x.split('/')[-1].split('_')[0])
    prediction_df['true_class'] = prediction_df['img_path'].apply(lambda x: x.split('/')[-2])
    prediction_df['name'] = prediction_df['img_path'].apply(lambda x: '.'.join('_'.join(x.split('/')[-1].split('_')[1:]).split('.')[:-1]))

    # TODO: move this below to edges_df... part of main
    prediction_df['probability'] = (prediction_df['probability'] * 100).astype(int)
    prediction_df = prediction_df[prediction_df['probability'] > thr]

    # link 'name' which is the particular image to the predicted class 'description' with weight 'probability'
    # -- color should be partition - shape should be 'true' == 'class'
    prediction_df['source'] = prediction_df['name']
    prediction_df['target'] = prediction_df['description']
    prediction_df['weight'] = prediction_df['probability']
    prediction_df.loc[prediction_df['true_class']==prediction_df['class'], 'shape'] = 'square'
    prediction_df.loc[prediction_df['true_class']!=prediction_df['class'], 'shape'] = 'circle'
    prediction_df.loc[prediction_df['true_class']==prediction_df['class'], 'y'] = 2
    prediction_df.loc[prediction_df['true_class']!=prediction_df['class'], 'y'] = 1

    # There will be duplicates...drop the 'squares'  sort keep=first
    all_nodes_df = prediction_df[['name', 'y', 'shape']].sort_values('shape').drop_duplicates(subset='name', keep='first')
    # then append the target description nodes...
    all_nodes_df = all_nodes_df.append(prediction_df[['description']].rename(columns={'description':'name'})).drop_duplicates(subset='name', keep='first')
    # 
    manual_y_df = all_nodes_df.copy().fillna(4).rename(columns={'name':'im_id'}) # , 'shape':'y'})
    prediction_df['img_name'] = prediction_df['img_path'].apply(lambda x: x.split('/')[-1])
    manual_y_df = manual_y_df.merge(prediction_df[['name', 'img_name']].drop_duplicates(), left_on='im_id', right_on='name', how='left')

    edges_df = prediction_df[['source', 'target', 'weight']].copy().reset_index(drop=True)

    return edges_df, manual_y_df   # TODO: three inputs, edges_df, labels_df

def generate_tree(graph, part_num, nodes):
    print('generate_tree: nodes', len(nodes)) 
    part_graph = nx.subgraph(graph, nodes).copy() # do we need the copy??
    # if len(part_graph.nodes()) < 3:  # not interesting...
        # return
    partition = community_louvain.best_partition(part_graph)
    sub_partition_df = pd.DataFrame([partition]).T.reset_index().rename(columns={'index':'node', 0:'part'})
    print('generate_tree: parts', sub_partition_df['part'].nunique())
    sub_graph = subgraph_sample(sub_partition_df, 500, part_graph)
    nodes_df, nodes_list = create_nodes_df(sub_partition_df, sub_graph, manual_y_df)
    links_df, edges_list = create_links_df(sub_graph)
    # links_df['part_num'] = part_num
    sub_partition_df['part_num'] = part_num
    # all_links.append(sub_partition_df)
    create_jsons(nodes_list, nodes_df, edges_list, model_name, part_num, thr)

    if (sub_partition_df['part'].nunique() < 2):  # done...
        return
    for sub_part_num, part_df in sub_partition_df.groupby('part'):
        part_df['sub_part_num'] = str(part_num) + 'p' + str(sub_part_num)
        all_links.append(part_df)
        generate_tree(graph, str(part_num) + 'p' + str(sub_part_num), part_df['node'])



"""
from vgg_graph import get_data 
from vgg_graph import subgraph_sample 
from vgg_graph import create_nodes_df 
from vgg_graph import create_links_df 
from vgg_graph import create_jsons 
"""

thr = 0 # TODO: needs to move inside...
def create_semantic_graph():
    links_df = pd.read_parquet('links_df.' + model_name + '.parq')
    prediction_df = pd.read_parquet('prediction_df.' + model_name + '.parq')
    links_df = links_df[links_df['node'].isin(prediction_df['description'].drop_duplicates())].reset_index(drop=True)
    edges_df = links_df.rename(columns={'node':'source', 'sub_part_num':'target'})[['source', 'target']]
    edges_df['len'] = edges_df['target'].str.len()
    edges_df = edges_df.sort_values('len').drop_duplicates('source', keep='last') # keep the deeper description, not the abstract..
    # flip source/target for DAG
    edges_df = edges_df.rename(columns={'source':'target', 'target':'source'})[['source', 'target']]

    # add edges between part and its children...
    add_edges = links_df[['part_num', 'sub_part_num']].drop_duplicates().copy().reset_index(drop=True)
    add_edges = add_edges.rename(columns={'part_num':'source', 'sub_part_num':'target'})
    edges_df = edges_df.append(add_edges)

    semantic_graph = nx.from_pandas_edgelist(edges_df[['source', 'target']], create_using=nx.DiGraph())
    return semantic_graph

def create_tree_data(data_name, semantic_graph):
    name = '-1'
    parent = 'null'
    children = get_level(semantic_graph, name, parent)
    tree_data = [{'name': name, 'parent': parent, 'children': children}]

    # create tree json file
    store_json = tree_data
    with open("./js/" + data_name + "_thr_" + str(thr) + "_tree.json" , "w") as outfile:
        json.dump(store_json, outfile)

def get_level(semantic_graph, name, parent):
    level_data = nx.descendants_at_distance(semantic_graph, name, 1)
    # nx.bfs_tree(semantic_graph, '-1')
    children = []
    for n in level_data:
        node = {'name': n, 'parent': name, 'children': get_level(semantic_graph, n, name)}
        children.append(node)
    return children

def compare_semantic_maps():
    # TODO: compare vgg vs resnet semantic maps
    links = []
    for model_name in ['vgg', 'resnet50']:
        links_df = pd.read_parquet('links_df.' + model_name + '.parq')
        prediction_df = pd.read_parquet('prediction_df.' + model_name + '.parq')
        links_df = links_df[links_df['node'].isin(prediction_df['description'].drop_duplicates())].reset_index(drop=True)
        edges_df = links_df.rename(columns={'node':'source', 'sub_part_num':'target'})[['source', 'target']]
        edges_df['len'] = edges_df['target'].str.len()
        edges_df = edges_df.sort_values('len').drop_duplicates('source', keep='last') # keep the deeper description, not the abstract..
        links.append(edges_df.drop('len', axis=1))


    # need to iterate up the tree, start at bottom, shared we pop off..
    done = []
    for offset in range(1, 5):  # depths of tree....
        edges0_df = links[0].copy()
        edges1_df = links[1].copy()
        edges0_df['target'] = edges0_df['target'].apply(lambda x: 'p'.join(x.split('p')[:-offset]))
        edges1_df['target'] = edges1_df['target'].apply(lambda x: 'p'.join(x.split('p')[:-offset]))
        # there are the nodes that are not shared...for now just drop them...
        links_df = edges0_df.merge(edges1_df, on='source', how='inner')
        if len(done) > 0:
            done_df = pd.concat(done)
            links_df = links_df[~links_df['source'].isin(done_df['source_y'])]
        count_df = links_df.groupby(['target_x', 'target_y']).count().reset_index()
        count_df = count_df[count_df['source'] > 1]
        count_df = count_df.merge(links_df, on=['target_x','target_y'])
        done.append(count_df[['source_y', 'target_y']].copy())

    done_df = pd.concat(done)
    # flip source/target for DAG
    done_df = done_df.rename(columns={'source_y':'target', 'target_y':'source'})[['source', 'target']]
    done_df = done_df[done_df['source'] != '']
    edges1_df = links[1].copy()  # we took _y from the merge...
    edges1_df = edges1_df.rename(columns={'source':'target', 'target':'source'})[['source', 'target']]
    final_edges_df = done_df.append(edges1_df).reset_index(drop=True)
    # duplicates now must be from the merge, and we merged the done before the orig, so keep first 
    final_edges_df = final_edges_df.drop_duplicates('target', keep='first') 
    edge_filter = ~final_edges_df['target'].isin(done_df['target'])
    final_edges_df.loc[edge_filter, 'target'] = 'lone_' + final_edges_df.loc[edge_filter]['target']

    add_edges = final_edges_df['source'].apply(lambda x: x == 'p'.join(x.split('p')[:2]))
    add_df = final_edges_df[add_edges][['source']].copy().reset_index(drop=True)
    add_df['target'] = '-1'
    add_df = add_df.rename(columns={'source':'target', 'target':'source'})[['source', 'target']]
    final_edges_df = final_edges_df.append(add_df).reset_index(drop=True)

    semantic_graph = nx.from_pandas_edgelist(final_edges_df[['source', 'target']], create_using=nx.DiGraph())
    create_tree_data('merged', semantic_graph)



# model_name = 'top_5_wrong_convnext'
# # model_name = 'vgg'
# all_links = []
# if __name__ == '__main__':
#     edges_df, manual_y_df = get_data()
#     graph = nx.from_pandas_edgelist(edges_df)
#     generate_tree(graph, '-1', graph.nodes())
#     links_df = pd.concat(all_links)
#     links_df.columns = [str(x) for x in links_df.columns]
#     # links_df.to_csv('links_df.csv')
#     links_df.to_parquet('links_df.' + model_name + '.parq')
#     semantic_graph = create_semantic_graph()
#     create_tree_data(model_name, semantic_graph)

# compare_semantic_maps()