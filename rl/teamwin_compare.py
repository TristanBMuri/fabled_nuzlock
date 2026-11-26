"""D19 bet check: does a team model trained on non-boss trainers (D19) judge boss teams better than one
trained on five bosses (D18)? Same held-out set for both: every logged box of Lt. Surge, Giovanni, Sabrina,
Pryce and Clair, and their same-box pairs (D19 --alt variants included).

    uv run python rl/teamwin_compare.py [win|utility]
"""
import sys, json, random, glob
import numpy as np, torch
sys.path.insert(0, '.')
from rl.fit_teamwin import load, features, boss_name, BOSSES, NAMES, spearman
from rl.fit_eval import fit
torch.manual_seed(0)
label = sys.argv[1] if len(sys.argv) > 1 else 'win'
rows = load(sorted(glob.glob('results/rr_*.events.jsonl*')))
X = np.array([features(e) for e in rows], dtype=np.float32)
Y = np.array([e['win'] * (1 - 0.5 * e['deaths_per_mon']) if label == 'utility' else e['win'] for e in rows], dtype=np.float32)
name = np.array([boss_name(e['boss']) for e in rows])
H = np.isin(name, ['LT. SURGE', 'GIOVANNI', 'SABRINA', 'PRYCE', 'CLAIR'])  # the D18 held-out bosses, every run
trains = {'d18 (Brock..Blaine boxes)': np.isin(name, ['BROCK', 'MISTY', 'ERIKA', 'KOGA', 'BLAINE']),
          'd19 (non-boss trainers)': ~np.isin(name, list(BOSSES)),
          'd18+d19': np.isin(name, ['BROCK', 'MISTY', 'ERIKA', 'KOGA', 'BLAINE']) | ~np.isin(name, list(BOSSES))}
groups = {}
for i, e in enumerate(rows):
    if H[i]:
        groups.setdefault((e['boss'], json.dumps(e['box']), json.dumps(e['cfg'])), []).append(i)
pairs = [(i, j) for idx in groups.values() for a, i in enumerate(idx) for j in idx[a + 1:] if Y[i] != Y[j] and rows[i]['sets'] != rows[j]['sets']]
print(f'label {label}; held-out boxes {H.sum()}, same-box pairs {len(pairs)}')
for tag, tr in trains.items():
    rng = random.Random(1); vb = {}
    val = np.array([vb.setdefault((e['boss'], json.dumps(e['box'])), rng.random() < 0.2) for e in rows]) & tr
    t = tr & ~val
    mu, sd = X[t].mean(0), X[t].std(0) + 1e-6
    T = lambda m: (torch.tensor((X[m] - mu) / sd), torch.tensor(Y[m]))
    (x, y), (xv, yv) = T(t), T(val)
    lin = fit(torch.nn.Linear(len(NAMES), 1), x, y, xv, yv)
    with torch.no_grad():
        p = torch.sigmoid(lin(torch.tensor((X - mu) / sd)).squeeze(-1)).numpy()
    acc = np.mean([(p[i] - p[j]) * (Y[i] - Y[j]) > 0 for i, j in pairs])
    big = [(i, j) for i, j in pairs if abs(Y[i] - Y[j]) >= 0.5]
    accb = np.mean([(p[i] - p[j]) * (Y[i] - Y[j]) > 0 for i, j in big])
    rc = spearman(p[H], Y[H], np.array([e['boss'] for e in rows])[H])
    print(f'  {tag:<28} train boxes {t.sum():>6}: same-box picks better {acc:.1%} (pairs with a >=50-pt gap: {accb:.1%}, n={len(big)}), rank corr {rc:+.3f}')
