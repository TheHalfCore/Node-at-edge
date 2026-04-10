def estimate_rf_memory(n_trees, n_leaves, n_classes,
                       bytes_per_pointer=4,
                       bytes_threshold=4,
                       bytes_feature=4,
                       bytes_per_prob=4):
    """
    Estimate memory usage of a Random Forest.

    Parameters:
    - n_trees: number of trees
    - n_leaves: number of leaves per tree
    - n_classes: number of classes
    - bytes_*: adjust if using float64 (8 bytes) etc.

    Returns:
    - memory in bytes, KB, MB
    """

    # Internal nodes (binary tree)
    n_internal_nodes = n_leaves - 1

    # Memory per internal node
    node_size = (
        bytes_per_pointer * 2 +  # left + right
        bytes_threshold +
        bytes_feature
    )

    # Memory per leaf
    leaf_size = n_classes * bytes_per_prob

    # Memory per tree
    memory_per_tree = (
        n_internal_nodes * node_size +
        n_leaves * leaf_size
    )

    # Total forest memory
    total_memory = n_trees * memory_per_tree

    return {
        "bytes": total_memory,
        "KB": total_memory / 1024,
        "MB": total_memory / (1024 ** 2)
    }

if __name__ == "__main__":
    # n_trees = 8
    # max_depth = 2
    n_trees = int(input("Enter number of trees: "))
    max_depth = int(input("Enter max depth: "))
    n_classes = int(input("Enter number of classes: "))
    n_leaves = 2**max_depth
    # n_classes = 17
    print(f"N_trees: {n_trees}, n_leaves: {n_leaves}, n_classes: {n_classes}")
    size = estimate_rf_memory(n_trees, n_leaves, n_classes)
    print(f"total Size: {size}")