import argparse
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import os

CONN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connections_princeton.csv")
NEURONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurons.csv")


def load_adjacency():
    df = pd.read_csv(
        CONN,
        dtype={
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "syn_count": "int32",
        },
        usecols=["pre_root_id", "post_root_id", "syn_count"],
    )
    adj = defaultdict(list)
    for row in df.itertuples(index=False, name="Edge"):
        adj[row.pre_root_id].append((row.post_root_id, row.syn_count))
    return adj


def load_annotations():
    df = pd.read_csv(
        NEURONS,
        dtype={"Root ID": "int64"},
        usecols=[
            "Root ID",
            "Super Class",
            "Class",
            "Function",
            "Body Part",
            "Nerve",
            "Primary Cell Type",
        ],
    )
    return df.set_index("Root ID")


def find_hub(adj, annotations, top_k=20):
    out_syn = {n: sum(s for _, s in edges) for n, edges in adj.items()}
    out_deg = {n: len(edges) for n, edges in adj.items()}
    rows = []
    for n in out_syn:
        a = annotations.loc[n] if n in annotations.index else None
        rows.append(
            (
                n,
                out_syn[n],
                out_deg[n],
                a["Class"] if a is not None else "",
                a["Function"] if a is not None else "",
                a["Super Class"] if a is not None else "",
            )
        )
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nTop {top_k} neurons by total output synapses:")
    print(f"{'root_id':<22} {'out_syn':>10} {'out_deg':>8}  {'class':<25} {'function':<14} {'super_class'}")
    for r in rows[:top_k]:
        print(f"{r[0]:<22} {r[1]:>10} {r[2]:>8}  {r[3]:<25} {r[4]:<14} {r[5]}")


def suggest_starts(adj, annotations, k=8):
    print(
        "\nThat neuron is a dead end. Real neurons to start from "
        "(sensory inputs, by total output synapses):",
        file=sys.stderr,
    )
    out_syn = {n: sum(s for _, s in edges) for n, edges in adj.items()}
    mask = annotations["Class"].fillna("").str.contains(
        "olfactory_receptor|photoreceptor|chordotonal|gustatory|bristle",
        case=False,
        na=False,
    )
    cands = annotations[mask].index
    rows = sorted(
        (
            (n, out_syn[n], annotations.loc[n]["Class"], annotations.loc[n]["Primary Cell Type"])
            for n in cands
            if n in out_syn
        ),
        key=lambda r: r[1],
        reverse=True,
    )
    print(f"{'root_id':<22} {'out_syn':>8}  {'class':<28} {'type'}", file=sys.stderr)
    for n, s, cls, t in rows[:k]:
        print(f"{n:<22} {s:>8}  {cls:<28} {t}", file=sys.stderr)


def propagate(adj, start, initial_signal=1000.0, max_hops=None):
    active = {start}
    strength = {start: initial_signal}
    level = {start: 0}
    parent = {}
    order = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        lvl = level[node]
        if max_hops is not None and lvl >= max_hops:
            continue
        for post, syn in adj.get(node, ()):
            if post not in active:
                active.add(post)
                strength[post] = syn
                level[post] = lvl + 1
                parent[post] = node
                queue.append(post)
            else:
                strength[post] += syn
    return active, strength, level, parent, order


def main():
    ap = argparse.ArgumentParser(description="Signal propagation over the fly connectome")
    ap.add_argument("--start", type=int, default=None, help="Starting neuron root ID")
    ap.add_argument("--initial", type=float, default=1000.0, help="Initial signal strength")
    ap.add_argument("--max-hops", type=int, default=None, help="Stop after N synaptic hops")
    ap.add_argument("--top", type=int, default=15, help="How many top active neurons to show")
    ap.add_argument("--hubs", action="store_true", help="Just list top hub neurons and exit")
    args = ap.parse_args()

    print("Loading neurons...")
    annotations = load_annotations()
    print("Loading connections and building graph...")
    adj = load_adjacency()
    print(f"Graph: {len(adj)} neurons with outgoing edges")

    if args.hubs:
        find_hub(adj, annotations)
        return

    start = args.start
    if start is None:
        out_syn = {n: sum(s for _, s in edges) for n, edges in adj.items()}
        start = max(out_syn, key=out_syn.get)
        a = annotations.loc[start] if start in annotations.index else None
        print(f"No --start given; using strongest output hub {start}")
        if a is not None:
            print(
                f"  ({a['Class']} | {a['Function']} | {a['Super Class']} | {a['Body Part']} | {a['Primary Cell Type']})"
            )
    else:
        has_out = start in adj
        has_in = any(
            p == start for edges in adj.values() for p, _ in edges
        )
        if not has_out:
            if has_in:
                print(
                    f"NOTE: {start} has NO outgoing synapses — it is an output target only. "
                    f"It receives input but fires to no one, so the signal dies here.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"NOTE: {start} has NO connections at all (isolated fragment). "
                    f"Signal cannot travel anywhere.",
                    file=sys.stderr,
                )
            suggest_starts(adj, annotations)
        a = annotations.loc[start] if start in annotations.index else None
        if a is not None:
            print(f"Starting neuron {start}: ({a['Class']} | {a['Function']} | {a['Super Class']} | {a['Primary Cell Type']})")

    print(f"\nInjecting signal={args.initial} into neuron {start}...")
    active, strength, level, parent, order = propagate(adj, start, args.initial, args.max_hops)

    total_pre = len(adj)
    total_known = len(annotations)
    n_active = len(active)
    n_known = sum(1 for n in active if n in annotations.index)
    max_hop = max(level.values())
    print(f"Active neurons: {n_active}  (of {total_known} annotated / {total_pre} in graph)")
    print(f"Synaptic hops until saturation: {max_hop}")
    print(f"Signal reached {100.0 * n_active / total_known:.1f}% of all known neurons")

    active_df = pd.DataFrame(
        {
            "root_id": list(active),
            "strength": [strength[n] for n in active],
            "hops": [level[n] for n in active],
        }
    ).sort_values("strength", ascending=False)
    annotated = active_df.merge(
        annotations, left_on="root_id", right_index=True, how="left"
    )

    hop_counts = pd.Series(level).value_counts().sort_index()
    print("\nWave size per synaptic hop:")
    print(hop_counts.to_string())

    direct = active_df[active_df["hops"] == 1].merge(
        annotations, left_on="root_id", right_index=True, how="left"
    )
    print(f"\nDirect postsynaptic targets of {start} (hop 1): {len(direct)}")
    if len(direct):
        print(
            direct[
                ["root_id", "strength", "Class", "Primary Cell Type", "Function", "Super Class"]
            ]
            .sort_values("strength", ascending=False)
            .head(args.top)
            .to_string(index=False)
        )

    print(f"\nTop {args.top} most active neurons:")
    cols = ["root_id", "strength", "hops", "Class", "Function", "Super Class", "Body Part", "Primary Cell Type"]
    print(annotated[cols].head(args.top).to_string(index=False))

    print("\nActivation by Class (top 15):")
    cc = annotated.groupby("Class").size().sort_values(ascending=False)
    print(cc.head(15).to_string())

    print("\nActivation by Function (top 10):")
    ff = annotated.groupby("Function").size().sort_values(ascending=False)
    print(ff.head(10).to_string())

    unann = annotated["Class"].isna().sum()
    print(f"\nUnannotated active neurons: {unann}")

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_neurons.csv")
    annotated.to_csv(out_csv, index=False)
    print(f"Full active-neuron table written to {out_csv}")


if __name__ == "__main__":
    main()
