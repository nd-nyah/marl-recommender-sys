class MetricTracker:

    def __init__(self):
        self.store = {}

    def log(self, key, value):
        if key not in self.store:
            self.store[key] = []
        self.store[key].append(value)

    def summary(self):
        return {
            k: sum(v) / len(v)
            for k, v in self.store.items()
        }