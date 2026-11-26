"""Team-judge comparisons on fixed held-out boss sets (D19 bet check, D23 data).

Training sets (non-boss trainers only, linear model, fit_teamwin features): D18 (five bosses' boxes), D19 (the
D19 data run: greedy six + swaps, 6 battles at 20,000 iterations), D23 (the D23 data run: cover six + swaps +
random sixes, 16 battles at 2,000 iterations), D19+D23. Held-out sets: (1) every logged box of Lt. Surge,
Giovanni, Sabrina, Pryce and Clair outside the D23 run (20k labels, 6 battles), (2) every boss box of the D23
run (gym leaders, Giovanni, E4, Lance; 2k labels, 16 battles), (3) the same without the random sixes (the
chooser's six vs its swaps: the builder's real question). Reported: same-box pairs where the model picks
the measured better team, and the within-boss rank correlation.

    uv run python rl/teamwin_compare.py [win|utility]
"""
import sys, json, random, glob
import numpy as np, torch
sys.path.insert(0, '.')
from rl.fit_teamwin import load, features, boss_name, BOSSES, NAMES, spearman
from rl.fit_eval import fit
torch.manual_seed(0)
label = sys.argv[1] if len(sys.argv) > 1 else 'win'
D23 = 'results/rr_d23_data.events.jsonl.xz'
old = load(sorted(p for p in glob.glob('results/rr_*.events.jsonl*') if p != D23))
new = load([D23]) if glob.glob(D23) else []
rows = old + new
src = np.array(['old'] * len(old) + ['d23'] * len(new))
X = np.array([features(e) for e in rows], dtype=np.float32)
Y = np.array([e['win'] * (1 - 0.5 * e['deaths_per_mon']) if label == 'utility' else e['win'] for e in rows], dtype=np.float32)
name = np.array([boss_name(e['boss']) for e in rows])
boss = np.array([e['boss'] for e in rows])
is_boss = np.isin(name, list(BOSSES))
five = ['LT. SURGE', 'GIOVANNI', 'SABRINA', 'PRYCE', 'CLAIR']
run19 = np.array([e['boss'].startswith('#') for e in rows]) & (src == 'old')  # the D19 run keys trainers by order index
trains = {'d18 (Brock..Blaine boxes)': np.isin(name, ['BROCK', 'MISTY', 'ERIKA', 'KOGA', 'BLAINE']) & (src == 'old'),
          'd19 (non-boss, D19 run)': ~is_boss & run19,
          'd23 (non-boss, D23 run)': ~is_boss & (src == 'd23'),
          'd19+d23': ~is_boss & (run19 | (src == 'd23'))}
helds = {'old held-out (5 bosses, 20k x6)': np.isin(name, five) & (src == 'old'),
         'D23 bosses (2k x16)': is_boss & (src == 'd23'),
         'D23 bosses, no random sixes': is_boss & (src == 'd23') & ~np.array([bool(e.get('rand')) for e in rows])}


def same_box_pairs(H):
    groups = {}
    for i in np.flatnonzero(H):
        e = rows[i]
        groups.setdefault((src[i], e['boss'], json.dumps(e['box']), json.dumps(e['cfg'])), []).append(i)
    return [(i, j) for idx in groups.values() for a, i in enumerate(idx) for j in idx[a + 1:]
            if Y[i] != Y[j] and rows[i]['sets'] != rows[j]['sets']]


print(f'label {label}; rows old {len(old)}, D23 {len(new)}')
for tag, H in helds.items():
    pairs = same_box_pairs(H)
    if not pairs:
        continue
    big = [(i, j) for i, j in pairs if abs(Y[i] - Y[j]) >= 0.5]
    print(f'{tag}: {H.sum()} boxes, {len(pairs)} same-box pairs ({len(big)} with a >=50-pt gap)')
    for ttag, tr in trains.items():
        if tr.sum() < 50:
            continue
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
        accb = np.mean([(p[i] - p[j]) * (Y[i] - Y[j]) > 0 for i, j in big]) if big else float('nan')
        print(f'  {ttag:<26} train {t.sum():>6}: picks better {acc:.1%} (>=50-pt gap {accb:.1%}), rank corr {spearman(p[H], Y[H], boss[H]):+.3f}')
