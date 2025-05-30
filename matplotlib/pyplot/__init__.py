def subplots(*args, **kwargs):
    class Fig:
        def savefig(self, *a, **k):
            pass
    return Fig(), []

def plot(*args, **kwargs):
    pass

def figure(*args, **kwargs):
    class Fig:
        def savefig(self, *a, **k):
            pass
    return Fig()
