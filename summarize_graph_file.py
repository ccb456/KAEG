# _*_ coding : utf-8 _*_
# @Author : 扶摇
import json
from pathlib import Path
from statistics import mean, median


def count_edges_in_doc(doc_graph):
    edge_count = 0
    for relation_dict in doc_graph:
        for targets in relation_dict.values():
            edge_count += len(targets)
    return edge_count


def summarize_graph_file(file_path):
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        graphs = json.load(f)

    edge_counts = [count_edges_in_doc(doc_graph) for doc_graph in graphs]

    total_docs = len(edge_counts)
    total_edges = sum(edge_counts)
    avg_edges = mean(edge_counts) if edge_counts else 0
    median_edges = median(edge_counts) if edge_counts else 0
    nonempty_docs = sum(1 for x in edge_counts if x > 0)
    nonempty_ratio = nonempty_docs / total_docs * 100 if total_docs else 0

    return {
        "docs": total_docs,
        "total_edges": total_edges,
        "avg_edges_per_doc": avg_edges,
        "median_edges_per_doc": median_edges,
        "docs_with_edges": nonempty_docs,
        "docs_with_edges_ratio": nonempty_ratio,
    }


def main():
    graph_files = {
        "DocRED Train": "dataset/DocRED/kg/train_annotated_graph.json",
        "DocRED Dev": "dataset/DocRED/kg/dev_graph.json",
        "DocRED Test": "dataset/DocRED/kg/test_graph.json",
        "Re-DocRED Train": "dataset/Re-DocRED/kg/train_revised_graph.json",
        "Re-DocRED Dev": "dataset/Re-DocRED/kg/dev_revised_graph.json",
        "Re-DocRED Test": "dataset/Re-DocRED/kg/test_revised_graph.json",
    }

    print(
        f"{'Split':<20}"
        f"{'Docs':>8}"
        f"{'Total Edges':>15}"
        f"{'Avg/Doc':>12}"
        f"{'Median/Doc':>14}"
        f"{'Docs w/ Edges':>16}"
        f"{'Ratio (%)':>12}"
    )

    for split, path in graph_files.items():
        stats = summarize_graph_file(path)
        print(
            f"{split:<20}"
            f"{stats['docs']:>8}"
            f"{stats['total_edges']:>15}"
            f"{stats['avg_edges_per_doc']:>12.2f}"
            f"{stats['median_edges_per_doc']:>14.2f}"
            f"{stats['docs_with_edges']:>16}"
            f"{stats['docs_with_edges_ratio']:>12.2f}"
        )


if __name__ == "__main__":
    main()