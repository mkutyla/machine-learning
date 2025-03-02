from treelib import Tree

tree = Tree()


def print_tree(dt):
    """
    Draws a DecisionTree in a readable format.

    Args:
        dt (DT.DecisionTree): a decision tree to draw
    """
    node = dt.root
    label = node.children_branches[0].label
    if ("<" in label) or (">" in label):
        tree.create_node(f"{node.label} {label}", f"{node}")
    else:
        tree.create_node(f"{node.label}", f"{node}")

    for branch in node.children_branches:
        __add_nodes(branch.child)

    filename = '../output/tree.txt'
    with open(filename, 'w') as _:
        tree.save2file(filename)


def __add_nodes(node):
    if node.is_leaf():
        if "<=" in node.parent_branch.label:
            tree.create_node(
                f"(1) {node.cls}", f"{node}", parent=f"{node.parent_branch.parent}"
            )
        elif ">" in node.parent_branch.label:
            tree.create_node(
                f"(0) {node.cls}", f"{node}", parent=f"{node.parent_branch.parent}"
            )
        else:
            label = node.parent_branch.label[1:]  # cutting off "="
            tree.create_node(
                f"({label}) {node.cls}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )

    else:
        label = node.children_branches[0].label
        if "<=" in node.parent_branch.label:
            tree.create_node(
                f"(1) {node.label} {label}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )
        elif ">" in node.parent_branch.label:
            tree.create_node(
                f"(0) {node.label} {label}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )
        else:
            label = node.parent_branch.label[1:]
            tree.create_node(
                f"({label}) {node.label}",
                f"{node}",
                parent=f"{node.parent_branch.parent}",
            )

    for branch in node.children_branches:
        __add_nodes(branch.child)
