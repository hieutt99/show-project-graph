import json
from explainer.GNNExplainer import GNNExplainer
from explainer.explainerVisualizer import *
from explainer.converter import *

from model.model import GCN
from model.dataloader import data_process, ToTensor

# todo: load pre-processed data
with open('../data/train_list.json', 'r') as fp:
    train_list = json.load(fp)

dataset = data_process(data_list=train_list[:], transform=ToTensor())

# todo: Load trained GCN
print("Loading trained GCN ...")
device = 'cpu'
model = GCN()
model.load_state_dict(torch.load('../model12.pth.tar', map_location=torch.device('cpu')))
# print(model)
print('Done!')

# todo: Demo
WANT_TO_TEST = [0]
TEST_ID = '../data/task1train_pro/X00016469612'
explainer = GNNExplainer(model, epochs=200, lr=0.01)

for i_test, sample in enumerate(dataset):
    if i_test not in WANT_TO_TEST: continue
    # todo: Load pre-processed data
    """
        V (Tensor): N x F  --> number of nodes x number of features 	(this case: N x 600)
        A (Tensor): L x N x N (this case: L = 4)
        edge_index (Tensor): 2 x number_of_edges
        edge_type (Tensor): number_of_edges
    """
    [V, A], label = sample
    edge_index, edge_type = convert_adj_to_edge_index(A)
    # print(edge_index)

    # todo: calculate predicted labels of nodes --> identify which nodes to be explained
    with torch.no_grad():
        pred = model.eval([V.to(device), A.to(device)])
    pred = pred[0].argmax(axis=1).cpu().numpy()

    indeces_of_nodes_to_explained = []
    values_of_nodes_to_explained = []
    for i in range(pred.shape[0]):
        if pred[i] != 0:
            indeces_of_nodes_to_explained.append(i)
            values_of_nodes_to_explained.append(pred[i])

    print("Nodes to be explained: ", indeces_of_nodes_to_explained)
    print("With respective label: ", values_of_nodes_to_explained)

    # todo: explain each nodes
    vis = imageVisualizer(TEST_ID)        # change the ID of the test here!!!
    edge_threshold = 0.9

    for node_id in indeces_of_nodes_to_explained:
        feature_mask, edge_mask = explainer.explain_node(node_id, V, A)
        """
            feature_mask (tensor): size = number_of_features; value in range[0,1]
            edge_mask (tensor): size = number_of_edge; value in range[0,1]
        """
        for z in range(edge_mask.size(0)):
            if edge_mask[z] > edge_threshold:
                # print(pred[node_id], int(edge_index[0][z]), int(edge_index[1][z]))
                vis.add_edge(pred[node_id], (int(edge_index[0][z]), int(edge_index[1][z])))

        print(feature_mask.min(), feature_mask.max())
        important_features = take_n_most_important(feature_mask)
        for feature_id in important_features: vis.add_feature(pred[node_id], feature_id)

    vis.draw_important_box(indeces_of_nodes_to_explained, values_of_nodes_to_explained)
    vis.draw_edges()
    vis.show()



