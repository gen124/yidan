import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.explain import GNNExplainer
import heapq

class TinyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden//4, heads=4, concat=True)
        self.conv2 = GATConv(hidden, hidden//4, heads=4, concat=True)
        self.readout = torch.nn.Linear(hidden, 1)
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        g = global_mean_pool(x, batch)
        out = self.readout(g).squeeze(-1)
        node_scores = self.readout(x).squeeze(-1)
        return out, node_scores

def pyg_data_from(nodes, feats, edges):
    x = torch.tensor(feats, dtype=torch.float32)
    if len(edges) == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long)
    else:
        pairs = [[u,v] for u,v,w in edges]
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

def fit_tiny_gnn_and_explain(nodes, feats, edges, model_prob, device="cpu", epochs=400, n_runs=1, noise_std=0.0):
    """
    Fit TinyGNN to a single sample probability and explain node/edge importance.
    For robustness on small-sample regimes we optionally run `n_runs` runs, each with small gaussian noise
    added to the node features and average the resulting importance masks.
    """
    data_base = pyg_data_from(nodes, feats, edges)
    data_base = data_base.to(device)

    all_node_masks = []
    all_edge_masks = []
    for run in range(max(1, int(n_runs))):
        # optionally add tiny noise to features for robustness
        if noise_std > 0:
            feats_noise = feats + np.random.randn(*feats.shape).astype(np.float32) * noise_std
            data = pyg_data_from(nodes, feats_noise, edges)
            data = data.to(device)
        else:
            data = data_base

        model = TinyGNN(in_channels=feats.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        target = torch.tensor([model_prob], dtype=torch.float32, device=device)
        model.train()
        for it in range(epochs):
            opt.zero_grad()
            out, _ = model(data.x, data.edge_index)
            loss = torch.nn.functional.mse_loss(torch.sigmoid(out), target)
            loss.backward(); opt.step()
        model.eval()
        # Use new torch-geometric GNNExplainer API: call explainer(model, x, edge_index, target=...)
        # Newer torch_geometric requires connecting an explainer_config + model_config.
        try:
            from torch_geometric.explain.config import ExplainerConfig, ModelConfig, ExplanationType, MaskType, ModelMode, ModelTaskLevel, ModelReturnType
            explainer = GNNExplainer(epochs=200)
            explainer.connect(
                ExplainerConfig(explanation_type=ExplanationType.model, node_mask_type=MaskType.attributes, edge_mask_type=None),
                ModelConfig(mode=ModelMode.binary_classification, task_level=ModelTaskLevel.graph, return_type=ModelReturnType.probs),
            )
            # GNNExplainer expects the passed model to return a single prediction tensor (graph-level)
            # Wrap the TinyGNN so it returns only the graph-level output used for loss calculation.
            class _ModelWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x, edge_index, batch=None):
                    out, _ = self.m(x, edge_index, batch)
                    return torch.sigmoid(out)

            wrapped = _ModelWrapper(model)
            explanation = explainer(wrapped, data.x, data.edge_index, target=target)
        except Exception:
            # fallback to older API
            explainer = GNNExplainer(epochs=200)
            wrapped = None
            class _ModelWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__(); self.m = m
                def forward(self, x, edge_index, batch=None):
                    out, _ = self.m(x, edge_index, batch); return torch.sigmoid(out)
            wrapped = _ModelWrapper(model)
            explanation = explainer(wrapped, data.x, data.edge_index, target=target)
        # Explanation object contains node_mask and edge_mask
        node_mask = explanation.node_mask
        edge_mask = getattr(explanation, 'edge_mask', None)
        nm = node_mask.detach().cpu().numpy()
        # reduce feature-wise masks to per-node scalar importances if needed
        if nm.ndim > 1:
            nm = nm.mean(axis=1)
        all_node_masks.append(nm)
        if edge_mask is not None:
            em = edge_mask.detach().cpu().numpy()
            if em.ndim > 1:
                em = em.mean(axis=0)
            all_edge_masks.append(em)
        else:
            all_edge_masks.append(None)

    # average results across runs
    if len(all_node_masks) == 1:
        node_imp = all_node_masks[0]
    else:
        node_imp = np.mean(np.stack(all_node_masks, axis=0), axis=0)

    if all_edge_masks and all_edge_masks[0] is not None:
        edge_stack = np.stack([e for e in all_edge_masks if e is not None], axis=0)
        edge_imp = np.mean(edge_stack, axis=0)
    else:
        edge_imp = None

    return node_imp, edge_imp

def path_pdm(node_scores, edges, topk=3):
    adj = {}
    for u,v,w in edges:
        adj.setdefault(u, []).append((v,w))
        adj.setdefault(v, []).append((u,w))
    N = len(node_scores)
    starts = np.argsort(-np.array(node_scores))[:3].tolist()
    results = []
    for s in starts:
        heap = [( -np.log(max(node_scores[s],1e-9)), s, [s] )]
        visited = {}
        while heap:
            cost, u, path = heapq.heappop(heap)
            if u in visited and visited[u] <= cost: continue
            visited[u] = cost
            results.append((cost, path))
            for v,w in adj.get(u,[]):
                new_cost = cost - np.log(max(node_scores[v],1e-9)) - np.log(max(w,1e-9))
                heapq.heappush(heap, (new_cost, v, path+[v]))
    seen=set(); out=[]
    for c,p in sorted(results, key=lambda x:x[0]):
        t=tuple(p)
        if t in seen: continue
        seen.add(t); out.append((c,p))
        if len(out)>=topk: break
    return out
