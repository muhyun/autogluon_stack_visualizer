# How to visualize the stack ensemble model of AutoGluon-Tabular

[AutoGluon Tabular](https://auto.gluon.ai/stable/index.html) trains a state-of-the-art tabular model using stacking ensemble model. Here I introduce a simple Graphviz based function to visualize the stack ensemble model.

```python
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
```

This is a stacked ensemble model trained by executing `fit()`.

![](./model_plain.png)


This is a new stacked ensemble model created by `fit_weighted_ensemble()` on the previous trained model. As seen here, this helps understading how AutoGluon stacks models to get the best model.

![](./model_weighted_ensemble.png)