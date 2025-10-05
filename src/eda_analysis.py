import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import os

# ==========================
# CONFIG
# ==========================
INPUT_FILE = "outputs/arxiv_cs_university.csv"
OUTPUT_DIR = "eda_results"
TOP_N_CATEGORIES = 5
TOP_AUTHORS = 500
SAMPLE_EDGES = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================
def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
    df["year"] = df["update_date"].dt.year

    print(f"Dataset contains {len(df)} records")
    print(df["university_match"].value_counts())

    # ==========================
    # 2.1 Publication Trends
    # ==========================
    print("Analyzing publication trends...")

    agg_publications = df.groupby(["university_match", "year"]).size().reset_index(name="count")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg_publications, x="year", y="count", hue="university_match", marker="o")
    plt.title("Publication Trends per University (2014–2024)")
    plt.xlabel("Year")
    plt.ylabel("Number of Publications")
    plt.grid(True)
    plt.legend(title="University")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/publication_trends.png")
    plt.close()

    print("Publication trends saved.")

    # ==========================
    # 2.2 Top Research Categories
    # ==========================
    print("Analyzing top research categories...")

    df["category_list"] = df["categories"].str.split()
    df_exploded = df.explode("category_list").dropna(subset=["category_list"])

    agg_categories = df_exploded.groupby(["university_match", "category_list"]).size().reset_index(name="count")
    top_categories = agg_categories.groupby("university_match").apply(
        lambda x: x.nlargest(TOP_N_CATEGORIES, "count")
    ).reset_index(drop=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_categories, x="count", y="category_list", hue="university_match")
    plt.title(f"Top {TOP_N_CATEGORIES} Research Categories per University")
    plt.xlabel("Number of Publications")
    plt.ylabel("Category")
    plt.legend(title="University")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_categories.png")
    plt.close()

    print("Top categories analysis saved.")

    # ==========================
    # 2.3 Co-authorship Network
    # ==========================
    print("Building co-authorship network (optimized)...")

    df["author_list"] = df["authors"].astype(str).str.split(",")

    author_counts = pd.Series(
        [a.strip() for lst in df["author_list"].dropna() for a in lst if a.strip()]
    ).value_counts()
    top_authors = set(author_counts.head(TOP_AUTHORS).index)

    edges = []
    for authors in df["author_list"].dropna():
        authors = [a.strip() for a in authors if a.strip()]
        authors = [a for a in authors if a in top_authors]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                edges.append((authors[i], authors[j]))

    print(f"Total filtered edges before sampling: {len(edges):,}")

    pd.DataFrame(edges, columns=["source", "target"]).to_csv(
        f"{OUTPUT_DIR}/coauthor_edges_full.csv", index=False
    )

    if len(edges) > SAMPLE_EDGES:
        edges = random.sample(edges, SAMPLE_EDGES)

    G = nx.Graph()
    G.add_edges_from(edges)

    print(f"Co-authorship Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (sampled)")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="blue", alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.title("Co-authorship Network (Sampled)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/coauthorship_network.png", dpi=300)
    plt.close()

    print("Co-authorship network saved.")

    # ==========================
    # 2.4 Summary Table
    # ==========================
    print("Generating summary table...")

    summary = df.groupby(["university_match", "year"]).size().reset_index(name="publication_count")
    summary.to_csv(f"{OUTPUT_DIR}/summary_publications.csv", index=False)

    print(f"Summary table saved to {OUTPUT_DIR}/summary_publications.csv")
    print("Comparative EDA complete!")


if __name__ == "__main__":
    main()
