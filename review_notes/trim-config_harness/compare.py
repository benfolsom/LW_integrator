import json, glob, collections, sys
dev, rev = {}, {}
for i in range(8):
    dev.update(json.load(open(f"out_dev_{i}.json"))); rev.update(json.load(open(f"out_rev_{i}.json")))
print("cases", len(dev), len(rev))
byentry = collections.Counter(); examples = collections.defaultdict(list)
for cid in dev:
    a, b = dev[cid], rev.get(cid)
    if b is None: byentry["missing"] += 1; continue
    for k in set(a) | set(b):
        if k == "_raw": continue
        if a.get(k) != b.get(k):
            byentry[k] += 1
            examples[k].append(cid)
print(byentry)
for k, ids in examples.items():
    print("==", k, len(ids)); 
    pref = collections.Counter(i.split("=")[0].rsplit(":",1)[0] if i.startswith("mut") else i.split(":")[0] for i in ids)
    keys = collections.Counter(i.split("=")[0].split(":",2)[-1] for i in ids)
    print("  bases", pref.most_common(8)); print("  keys", keys.most_common(40))
json.dump({k: v for k, v in examples.items()}, open("diffs.json", "w"))
