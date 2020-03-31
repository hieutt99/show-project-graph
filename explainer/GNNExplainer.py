import torch
from tqdm import tqdm
from math import sqrt
import queue
import numpy as np
from explainer.converter import *

EPS = 1e-15

class GNNExplainer(torch.nn.Module):

    coeffs = {
        'log_logits': 1.0,
        'edge_size': 0.005,
        'node_feat_size': 0.03,
        'edge_ent': 1.0,
        'node_feat_ent': 1.0,
        'lap': 0.5,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True, num_hop=2):
        super(GNNExplainer, self).__init__()
        """ Main args """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.edge_mask = None
        self.node_feature_mask = None

        """ Other args """
        self.num_hop = num_hop

    def __set_masks__(self, V, edge_index):
        (N, F), E = V.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))        # ???
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        #print("MIN, MAX, SUM, edge_mask: ", self.edge_mask.min(), self.edge_mask.max(), self.edge_mask.sum())


    def __subgraph__(self, x, V, A, edge_index, edge_type):
        """
            Create a subgraph with a k-hop subgraph around node "x" (x_th node) using BFS algorithms
            x: node id
            V: N x F
            edge_index: 2 x number_of_edges
        """
        N = V.size(0)
        num_edges = edge_index.size(1)
        Q = queue.Queue(maxsize=N)
        Q.put((x, 0))
        visited = np.zeros(N)
        visited[x] = 1
        hard_edge_mask = torch.zeros(num_edges).long()
        while not Q.empty():
            (u, depth) = Q.get()
            if depth >= self.num_hop: continue
            for i in range(num_edges):
                if u != int(edge_index[1][i]) and u != int(edge_index[0][i]): continue
                v = int(edge_index[0][i])
                if u == v: v = int(edge_index[1][i])
                hard_edge_mask[i] = True
                if visited[v] == 0:
                    visited[v] = 1
                    Q.put((v, depth + 1))

        subset = torch.from_numpy(np.asarray(np.where(visited == 1)).flatten())
        chosen_node = V[subset, :]

        list_edge_index = torch.from_numpy(np.asarray(np.where(hard_edge_mask == 1)).flatten())
        chosen_edge_index = edge_index[:, list_edge_index]
        chosen_edge_type = edge_type[list_edge_index]

        node_new_id = np.zeros(N)
        for i in range(subset.size(0)):
            node_new_id[subset[i]] = i

        for i in range(chosen_edge_index.size(1)):
            chosen_edge_index[0][i] = node_new_id[chosen_edge_index[0][i]]
            chosen_edge_index[1][i] = node_new_id[chosen_edge_index[1][i]]

        adj_mat_new = convert_edge_index_to_adj(chosen_edge_index, chosen_edge_type, chosen_node.size(0), A.size(0))

        #print("SUBSET: ", subset)
        #print(chosen_node.size())
        #print(chosen_edge_index.size())
        #print(hard_edge_mask.size())
        #print(chosen_edge_index)
        return chosen_node, adj_mat_new, chosen_edge_index, chosen_edge_type, hard_edge_mask

    def __loss__(self, node_idx, log_logits, pred_label, A_masked=None, num_edge=None):
        loss = self.coeffs['log_logits'] * -log_logits[node_idx, pred_label[node_idx]]
        #print("\nLog loss: ", self.coeffs['log_logits'] * -log_logits[node_idx, pred_label[node_idx]])

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        #print("Edge size loss: ", self.coeffs['edge_size'] * m.sum())

        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        #print("Edge entropy loss: ", self.coeffs['edge_ent'] * ent.mean())

        m = self.node_feat_mask.sigmoid()
        loss = loss + self.coeffs['node_feat_size'] * m.sum()
        #print("Feature size loss: ", self.coeffs['node_feat_size'] * m.sum())

        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        #print("Feature entropy loss: ", self.coeffs['node_feat_ent'] * ent.mean())
        # laplacian loss
        lap_loss = 0
        """
        for type in range(A_masked.size(0)):
            D = torch.diag(torch.sum(A_masked[type], 0))
            L = D - A_masked[type]
            pred_label_t = torch.tensor(pred_label, dtype=torch.float)
            lap_loss = lap_loss + (pred_label_t @ L @ pred_label_t) / num_edge
        loss = loss + self.coeffs['lap'] * lap_loss
        """
        #print("Lap loss: ", self.coeffs['lap'] * lap_loss)

        return loss

    def explain_node(self, node_idx, V, A):
        edge_index, edge_type = convert_adj_to_edge_index(A)
        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        # V = (n' x F), A = (n' x n'),  edge_index = (2 x new_number_of_edge),  hard_edge_mask = (old_number_of_edge)
        # IMPORTANT: After this, size of V, A, ... have changed !!! --> the mask size is small !!!
        V, A, edge_index, edge_type, hard_edge_mask = self.__subgraph__(node_idx, V, A, edge_index, edge_type)

        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model.eval([V, A])[0]  # because SOFTMAX
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(V, edge_index)
        self.to(V.device)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            #V_masked = V * self.node_feat_mask.view(1, -1).sigmoid()
            #A_masked = A
            #print(V_masked.size())
            #V_masked = V
            #A_masked = convert_edge_index_to_adj(edge_index, edge_type, A.size(1), A.size(0), use_mask_value=True, edge_mask=self.edge_mask.sigmoid())


            # USING REPARAMETRIZATION TRICK 
            feat_mask = torch.sigmoid(self.node_feat_mask)
            std_tensor = torch.ones_like(V, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(V, dtype=torch.float) - V
            z = torch.normal(mean=mean_tensor, std=std_tensor)
            V_masked = V + z * (1 - feat_mask)
            A_masked = convert_edge_index_to_adj(edge_index, edge_type, A.size(1), A.size(0), use_mask_value=True, edge_mask=self.edge_mask.sigmoid())
            #f (epoch == self.epochs or epoch == 1):
            #    print("V_masked = ", V_masked)


            log_logits = self.model.eval([V_masked, A_masked])[0]
            loss = self.__loss__(0, log_logits, pred_label, A_masked, num_edges)
            loss.backward()
            optimizer.step()


            if (epoch % 30 == 0):
                print('\n')
                print("Loss:  ", loss)
                #print("MIN, MAX, SUM, feature_mask: ", self.node_feat_mask.min(), self.node_feat_mask.max(), self.node_feat_mask.sum())

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        #print("MIN, MAX, SUM, feature_mask: ", self.node_feat_mask.min(), self.node_feat_mask.max(), self.node_feat_mask.sum())
        #print("MIN, MAX, SUM, edge_mask: ", self.edge_mask.min(), self.edge_mask.max(), self.edge_mask.sum())

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        #node_feat_mask = self.node_feat_mask.detach()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        #edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        counter = 0
        for i in range(hard_edge_mask.size(0)):
            if hard_edge_mask[i] == 1:
                edge_mask[i] = self.edge_mask.detach().sigmoid()[counter]
                #edge_mask[i] = self.edge_mask.detach()[counter]
                counter += 1

        return node_feat_mask, edge_mask
