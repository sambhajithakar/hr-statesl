"""Descriptive statistics of the HR-StateSL v1.0 curated edge resource.

Every number printed here is counted directly from HR-StateSL-edges-v1.0.csv.
No value is assumed, imputed or simulated.
"""
import csv, json, collections, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / 'HR-StateSL-edges-v1.0.csv')))

def tally(key):
    c = collections.Counter()
    for r in rows:
        for v in r[key].split(';'):
            c[v.strip()] += 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

stats = {
    'n_edges': len(rows),
    'n_distinct_gene_a': len({r['gene_a'] for r in rows}),
    'by_evidence_class': tally('evidence_class'),
    'by_causal_support': tally('causal_support'),
    'by_direction': tally('direction'),
    'by_hr_state': tally('hr_state'),
    'by_ctdna_detectability': tally('ctdna_detectability'),
}

# Interventional versus non-interventional
interventional = [r for r in rows if r['evidence_class'].startswith(('E1', 'E2'))]
stats['n_interventional'] = len(interventional)
stats['frac_interventional'] = round(len(interventional) / len(rows), 4)
low = [r for r in rows if r['causal_support'] == 'low']
stats['n_low_support'] = len(low)
stats['frac_low_support'] = round(len(low) / len(rows), 4)

# Target-like edges only (an SL or anti-SL claim, i.e. something a pipeline could rank)
TARGETLIKE = {'SL', 'anti-SL', 'SL-combination'}
target_rows = [r for r in rows if r['direction'] in TARGETLIKE]
stats['n_target_like_edges'] = len(target_rows)
stats['target_like_by_causal_support'] = dict(
    collections.Counter(r['causal_support'] for r in target_rows))
stats['frac_target_like_low_support'] = round(
    sum(r['causal_support'] == 'low' for r in target_rows) / len(target_rows), 4)

# Sign discordance: same gene_a appearing with both an SL and an anti-SL claim
dirs_by_gene = collections.defaultdict(set)
states_by_gene = collections.defaultdict(set)
for r in target_rows + [r for r in rows if r['direction'] == 'anti-SL']:
    dirs_by_gene[r['gene_a']].add(r['direction'])
    states_by_gene[r['gene_a']].update(s.strip() for s in r['hr_state'].split(';'))
discordant = sorted(g for g, d in dirs_by_gene.items()
                    if 'SL' in d and 'anti-SL' in d)
stats['sign_discordant_genes'] = discordant
stats['n_sign_discordant_genes'] = len(discordant)
stats['discordant_states'] = {g: sorted(states_by_gene[g]) for g in discordant}

# Genes carrying an anti-SL claim in at least one state: these are the edges a
# state-agnostic pipeline can surface as candidate targets by mistake.
anti = sorted({r['gene_a'] for r in rows if r['direction'] == 'anti-SL'})
stats['anti_sl_genes'] = anti
stats['n_anti_sl_genes'] = len(anti)

# Breast-specific versus transferred evidence
breast_terms = ('breast', 'TNBC')
non_breast = [r['edge_id'] for r in rows
              if any(t in r['notes'] for t in ('pancreatic', 'ovarian', 'hepatocellular'))]
stats['n_edges_flagged_non_breast_origin'] = len(non_breast)
stats['edges_flagged_non_breast_origin'] = non_breast

# Provenance
stats['n_distinct_sources'] = len({r['provenance_doi'] for r in rows})
stats['n_preprint_sourced'] = sum('preprint' in r['notes'].lower() for r in rows)

out = ROOT / 'results' / 'resource_stats.json'
out.write_text(json.dumps(stats, indent=2))
json.dump(stats, sys.stdout, indent=2)
print()
