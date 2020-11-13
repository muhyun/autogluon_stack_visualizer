from autogluon import TabularPrediction as task
import networkx as nx
from PIL import Image

def generate_model_visual(predictor, model_image_fname='model.png'):
    G = predictor._trainer.model_graph
    remove = [node for node,degree in dict(G.degree()).items() if degree < 1]
    G.remove_nodes_from(remove)
    root_node = [n for n,d in G.out_degree() if d==0]
    best_model_node = predictor.get_model_best()
    
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['label'] = 'Ensemble stack (Blue box is the best model)'
    A.graph_attr['labelloc']='t'

    A.graph_attr.update(rankdir='BT')
    A.node_attr.update(fontsize=10)
    A.node_attr.update(shape='rectangle')
    for node in A.iternodes():
        node.attr['label'] = f"{node.name}\nVal score: {float(node.attr['val_score']):.4f}"
        
        if node.name == best_model_node:
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = '#ff9900'
            node.attr['shape'] = 'box3d'
        elif nx.has_path(G, node.name, best_model_node):
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = '#ffcc00'
            
    A.draw(model_image_fname, format='png', prog='dot')

train_data = task.Dataset(file_path='./train.csv')

predictor = task.fit(train_data=train_data, label='y', time_limits=60, 
                     stack_ensemble_levels=2, num_bagging_folds=2)

generate_model_visual(predictor, './model_plain.png')

predictor.fit_weighted_ensemble()
generate_model_visual(predictor, './model_weighted_ensemble.png')

