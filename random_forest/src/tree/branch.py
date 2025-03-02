class Branch:

    def __init__(self, parent, child, label: str) -> None:
        self.parent = parent
        self.child = child
        self.label = label
