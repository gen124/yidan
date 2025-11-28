import os
import time
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import ResNet3D
from data_utils import PDVolDataset
from metrics import compute_metrics


def load_cfg(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def collate_batch(batch):
    # batch is a list of samples: {'volume': tensor (C,D,H,W), 'aal': np|None,'label':int,'id':str}
    vols = torch.stack([s['volume'] for s in batch], dim=0)
    labels = torch.tensor([s['label'] for s in batch], dtype=torch.float32)
    ids = [s['id'] for s in batch]
    # keep aal as list of numpy arrays or None
    aals = [s['aal'] for s in batch]
    return {'volume': vols, 'aal': aals, 'label': labels, 'id': ids}


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch['volume'].to(device)
        y = batch['label'].to(device)
        opt.zero_grad()
        out = model(x)  # returns logits shape (B,)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        logits = logits.view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward(); opt.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def validate(model, loader, device):
    model.eval()
    ys = []
    yps = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['volume'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits.view(-1)
            probs = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            ys.extend(y.detach().cpu().numpy().tolist())
            yps.extend(probs.detach().cpu().numpy().tolist())
    metrics = compute_metrics(ys, yps)
    metrics['val_loss'] = total_loss / max(1, n)
    return metrics


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--work-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (overrides config)')
    parser.add_argument('--synthetic', action='store_true', help='use synthetic dataset for quick smoke-test')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    cfg_train = cfg.get('train', {})
    device = args.device if args.device is not None else cfg_train.get('device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    seed = cfg_train.get('seed', 42)
    np.random.seed(seed); torch.manual_seed(seed)

    # dataset
    aug = cfg_train.get('augmentations', {})
    if args.synthetic:
        train_ds = PDVolDataset(synthetic=True, mode='train', input_size=tuple(cfg['data'].get('input_size', (192,192,128))), transform=aug)
        val_ds = PDVolDataset(synthetic=True, mode='val', input_size=tuple(cfg['data'].get('input_size', (192,192,128))))
    else:
        train_csv = cfg['data'].get('train_csv')
        val_csv = cfg['data'].get('val_csv')
        train_ds = PDVolDataset(manifest_csv=train_csv, mode='train', input_size=tuple(cfg['data'].get('input_size', (192,192,128))), transform=aug)
        val_ds = PDVolDataset(manifest_csv=val_csv, mode='val', input_size=tuple(cfg['data'].get('input_size', (192,192,128))))

    batch_size = int(cfg_train.get('batch_size', 2))
    # use top-level collate_batch so DataLoader can use multiprocessing safely
    # small-sample optimization: optional balanced sampler (helpful when dataset is tiny/imbalanced)
    if cfg_train.get('balanced_sampler', False):
        labels = [train_ds[i]['label'] for i in range(len(train_ds))]
        class_sample_count = np.array([labels.count(c) for c in sorted(set(labels))])
        # inverse frequency per sample
        weight_per_class = {c: 1.0 / (labels.count(c) + 1e-12) for c in set(labels)}
        weights = [weight_per_class[l] for l in labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_batch)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    model = ResNet3D(in_channels=cfg['model'].get('in_channels', 3), base_channels=cfg['model'].get('base_channels', 16), norm=cfg['model'].get('norm', 'group'), num_groups=int(cfg['model'].get('num_groups', 8)), dropout=float(cfg['model'].get('dropout', 0.0))).to(device)

    lr = float(cfg_train.get('lr', 1e-4))
    wd = float(cfg_train.get('weight_decay', 1e-5))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(cfg_train.get('max_epochs', 80))
    patience = int(cfg_train.get('patience', 10))

    ensure_dir(args.work_dir)
    best_val_auc = -1.0
    best_ck_path = os.path.join(args.work_dir, 'best_checkpoint.pth')
    last_ck_path = os.path.join(args.work_dir, 'last_checkpoint.pth')

    no_improve = 0
    t0 = time.time()
    for epoch in range(1, max_epochs+1):
        t1 = time.time()
        train_loss = train_epoch(model, train_loader, opt, device)
        val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t1

        print(f"Epoch {epoch}/{max_epochs}  train_loss={train_loss:.4f}  val_loss={val_metrics['val_loss']:.4f}  val_auc={val_metrics.get('auc', float('nan')):.4f}  time={elapsed:.1f}s")

        # save last
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'val_metrics': val_metrics}, last_ck_path)

        # best by AUC
        cur_auc = val_metrics.get('auc', float('nan'))
        if not np.isnan(cur_auc) and cur_auc > best_val_auc:
            best_val_auc = cur_auc
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'val_metrics': val_metrics}, best_ck_path)
            print('Saved new best checkpoint ->', best_ck_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'No improvement for {patience} epochs, stopping early.')
            break

    total_time = time.time() - t0
    print('Training finished. Best val AUC:', best_val_auc, 'Total time(s):', int(total_time))


if __name__ == '__main__':
    main()
