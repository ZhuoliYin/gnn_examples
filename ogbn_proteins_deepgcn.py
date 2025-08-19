import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter

# dataset = PygNodePropPredDataset('ogbn-proteins', root='./data')
# splitted_idx = dataset.get_idx_split()
# data = dataset[0]
# data.node_species = None
# data.y = data.y.to(torch.float)

# # Initialize features of nodes by aggregating edge features.
# row, col = data.edge_index
# data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# # Set split indices to masks.
# for split in ['train', 'valid', 'test']:
#     mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     mask[splitted_idx[split]] = True
#     data[f'{split}_mask'] = mask

# train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,
#                                 num_workers=5)
# test_loader = RandomNodeLoader(data, num_parts=5, num_workers=5)


# class DeeperGCN(torch.nn.Module):
#     def __init__(self, hidden_channels, num_layers):
#         super().__init__()

#         self.node_encoder = Linear(data.x.size(-1), hidden_channels)
#         self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

#         self.layers = torch.nn.ModuleList()
#         for i in range(1, num_layers + 1):
#             conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
#                            t=1.0, learn_t=True, num_layers=2, norm='layer')
#             norm = LayerNorm(hidden_channels, elementwise_affine=True)
#             act = ReLU(inplace=True)

#             layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
#                                  ckpt_grad=i % 3)
#             self.layers.append(layer)

#         self.lin = Linear(hidden_channels, data.y.size(-1))

#     def forward(self, x, edge_index, edge_attr):
#         x = self.node_encoder(x)
#         edge_attr = self.edge_encoder(edge_attr)

#         x = self.layers[0].conv(x, edge_index, edge_attr)

#         for layer in self.layers[1:]:
#             x = layer(x, edge_index, edge_attr)

#         x = self.layers[0].act(self.layers[0].norm(x))
#         x = F.dropout(x, p=0.1, training=self.training)

#         return self.lin(x)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
# evaluator = Evaluator('ogbn-proteins')


# def train(epoch):
#     model.train()

#     pbar = tqdm(total=len(train_loader))
#     pbar.set_description(f'Training epoch: {epoch:04d}')

#     total_loss = total_examples = 0
#     for data in train_loader:
#         optimizer.zero_grad()
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.edge_attr)
#         loss = criterion(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()

#         total_loss += float(loss) * int(data.train_mask.sum())
#         total_examples += int(data.train_mask.sum())

#         pbar.update(1)

#     pbar.close()

#     return total_loss / total_examples


# @torch.no_grad()
# def test():
#     model.eval()

#     y_true = {'train': [], 'valid': [], 'test': []}
#     y_pred = {'train': [], 'valid': [], 'test': []}

#     pbar = tqdm(total=len(test_loader))
#     pbar.set_description(f'Evaluating epoch: {epoch:04d}')

#     for data in test_loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.edge_attr)

#         for split in y_true.keys():
#             mask = data[f'{split}_mask']
#             y_true[split].append(data.y[mask].cpu())
#             y_pred[split].append(out[mask].cpu())

#         pbar.update(1)

#     pbar.close()

#     train_rocauc = evaluator.eval({
#         'y_true': torch.cat(y_true['train'], dim=0),
#         'y_pred': torch.cat(y_pred['train'], dim=0),
#     })['rocauc']

#     valid_rocauc = evaluator.eval({
#         'y_true': torch.cat(y_true['valid'], dim=0),
#         'y_pred': torch.cat(y_pred['valid'], dim=0),
#     })['rocauc']

#     test_rocauc = evaluator.eval({
#         'y_true': torch.cat(y_true['test'], dim=0),
#         'y_pred': torch.cat(y_pred['test'], dim=0),
#     })['rocauc']

#     return train_rocauc, valid_rocauc, test_rocauc


# for epoch in range(1, 1001):
#     loss = train(epoch)
#     train_rocauc, valid_rocauc, test_rocauc = test()
#     print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
#           f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')



import os
import sys
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter

class DeeperGCN(torch.nn.Module):
    def __init__(self, in_channels_x, in_channels_e, out_channels, hidden_channels=64, num_layers=28):
        super().__init__()
        self.node_encoder = Linear(in_channels_x, hidden_channels)
        self.edge_encoder = Linear(in_channels_e, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=(i % 3))
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)

def build_data(root='./data'):
    dataset = PygNodePropPredDataset('ogbn-proteins', root=root)
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)

    # Initialize node features by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

    # Masks
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask
    return data

def train_one_epoch(model, optimizer, criterion, loader, device, epoch):
    model.train()
    pbar = tqdm(total=len(loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.train_mask
        loss = criterion(out[mask], batch.y[mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(mask.sum())
        total_examples += int(mask.sum())
        pbar.update(1)
    pbar.close()
    return total_loss / max(total_examples, 1)

@torch.no_grad()
def evaluate(model, evaluator, loader, device, epoch):
    model.eval()
    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        for split in y_true.keys():
            mask = batch[f'{split}_mask']
            y_true[split].append(batch.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())
        pbar.update(1)
    pbar.close()

    def roc(split):
        return evaluator.eval({
            'y_true': torch.cat(y_true[split], dim=0),
            'y_pred': torch.cat(y_pred[split], dim=0),
        })['rocauc']

    return roc('train'), roc('valid'), roc('test')

def main():
    # On macOS/Windows, use spawn and guard worker creation.
    if sys.platform.startswith(('win', 'cygwin')) or sys.platform == 'darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)

    data = build_data(root='./data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeeperGCN(
        in_channels_x=data.x.size(-1),
        in_channels_e=data.edge_attr.size(-1),
        out_channels=data.y.size(-1),
        hidden_channels=64,
        num_layers=28
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    evaluator = Evaluator('ogbn-proteins')

    # Use >0 workers only under __main__ guard (we are now).
    # If you still run this inside a notebook, set workers to 0.
    use_workers = 5  # set to 0 if running in Jupyter
    train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,
                                    num_workers=use_workers, persistent_workers=(use_workers > 0))
    test_loader = RandomNodeLoader(data, num_parts=5, num_workers=use_workers,
                                   persistent_workers=(use_workers > 0))

    for epoch in range(1, 1001):
        loss = train_one_epoch(model, optimizer, criterion, train_loader, device, epoch)
        tr, va, te = evaluate(model, evaluator, test_loader, device, epoch)
        print(f'Epoch {epoch:04d} | Loss {loss:.4f} | Train {tr:.4f} | Val {va:.4f} | Test {te:.4f}')

if __name__ == '__main__':
    main()
