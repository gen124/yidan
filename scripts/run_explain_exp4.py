import os
import csv
import runpy
import sys

# ensure repo root is on sys.path so local imports in inference.py work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# load inference.py as a module namespace to avoid import path issues
inf_mod = runpy.run_path(os.path.join(repo_root, 'inference.py'))
run_inference_and_explain = inf_mod['run_inference_and_explain']

EXP_DIR = 'outputs/exp4'
CKPT = os.path.join(EXP_DIR, 'final_best.pth')
CFG = os.path.join(EXP_DIR, 'config.yaml')
MANIFEST = 'data/val_manifest.csv'
OUT_ROOT = os.path.join(EXP_DIR, 'explain_out')

os.makedirs(OUT_ROOT, exist_ok=True)
missing = []
errors = []
count=0
with open(MANIFEST, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 5:
            continue
        qsm_path, t1_path, aal_path, label, pid = row
        # skip non-PD cases: label expected '1' for PD
        try:
            is_pd = int(label) == 1
        except Exception:
            is_pd = False
        if not is_pd:
            continue

        if not os.path.exists(qsm_path):
            missing.append((pid, 'qsm', qsm_path))
            continue
        if t1_path.strip() == '' or not os.path.exists(t1_path):
            missing.append((pid, 't1', t1_path))
            continue
        if not os.path.exists(aal_path):
            # allow missing aal
            aal_path = None
        out_dir = os.path.join(OUT_ROOT, pid)
        os.makedirs(out_dir, exist_ok=True)
        try:
            print(f"Processing {pid} -> {out_dir}")
            res = run_inference_and_explain(CKPT, qsm_path, t1_path, aal_path, cfg_path=CFG, out_dir=out_dir)
            print(f"  Done: prob={res['prob']:.4f}")
            count+=1
        except Exception as e:
            print(f"  Error for {pid}: {e}")
            errors.append((pid, str(e)))

print('\nSummary:')
print(f'  Processed: {count}')
print(f'  Missing entries (skipped): {len(missing)}')
for m in missing[:10]:
    print('   ', m)
print(f'  Errors: {len(errors)}')
for e in errors[:10]:
    print('   ', e)
