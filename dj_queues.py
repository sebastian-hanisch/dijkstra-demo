"""Prioritätswarteschlangen für Dijkstra, alle mit denselben Zählern: `pushes` (neue Einträge), `decrease_keys` (Schlüssel eines vorhandenen Knotens gesenkt), `pops` (entnommen),
`work` (Schlüsselvergleiche; bei Dial: Eimerschritte). Alle liefern bei gleichen Eingaben dieselben Entfernungen - nur der Aufwand unterscheidet sich.

Schnittstelle: `push_or_decrease(node, key)`, `pop_min() -> (key, node)`, `len(queue)`."""

import heapq


class _Counters:
    def __init__(self):
        self.pushes = self.decrease_keys = self.pops = self.work = 0

    def as_dict(self):
        return {"pushes": self.pushes, "decrease_keys": self.decrease_keys, "pops": self.pops, "work": self.work}


class ArrayQueue(_Counters):
    """Ungeordnete Menge, Minimum durch lineare Suche: O(1) einfügen, O(Anzahl) entnehmen - insgesamt O(V²)."""

    def __init__(self, n=0, max_weight=None):
        super().__init__()
        self.key = {}

    def __len__(self):
        return len(self.key)

    def push_or_decrease(self, node, key):
        if node in self.key:
            self.decrease_keys += 1
        else:
            self.pushes += 1
        self.key[node] = key

    def pop_min(self):
        best_node, best_key = None, None
        for node, k in self.key.items():
            if best_node is None:
                best_node, best_key = node, k
            else:
                self.work += 1
                if k < best_key:
                    best_node, best_key = node, k
        del self.key[best_node]
        self.pops += 1
        return best_key, best_node


class BinaryHeapQueue(_Counters):
    """Binärheap mit Positionsfeld und echtem Decrease-Key: O(log n) je Operation."""

    def __init__(self, n=0, max_weight=None):
        super().__init__()
        self.heap, self.pos, self.key = [], {}, {}

    def __len__(self):
        return len(self.heap)

    def _less(self, i, j):
        self.work += 1
        return self.key[self.heap[i]] < self.key[self.heap[j]]

    def _swap(self, i, j):
        h = self.heap
        h[i], h[j] = h[j], h[i]
        self.pos[h[i]], self.pos[h[j]] = i, j

    def _up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if not self._less(i, parent):
                break
            self._swap(i, parent)
            i = parent

    def _down(self, i):
        n = len(self.heap)
        while True:
            left, right, small = 2 * i + 1, 2 * i + 2, i
            if left < n and self._less(left, small):
                small = left
            if right < n and self._less(right, small):
                small = right
            if small == i:
                return
            self._swap(i, small)
            i = small

    def push_or_decrease(self, node, key):
        if node in self.pos:
            self.decrease_keys += 1
            self.key[node] = key
            self._up(self.pos[node])
        else:
            self.pushes += 1
            self.key[node] = key
            self.heap.append(node)
            self.pos[node] = len(self.heap) - 1
            self._up(len(self.heap) - 1)

    def pop_min(self):
        node = self.heap[0]
        key = self.key.pop(node)
        last = self.heap.pop()
        del self.pos[node]
        if self.heap:
            self.heap[0] = last
            self.pos[last] = 0
            self._down(0)
        self.pops += 1
        return key, node


class LazyHeapQueue(_Counters):
    """Binärheap ohne Decrease-Key (`heapq`): ein gesenkter Schlüssel legt einen zweiten Eintrag an, der alte wird beim Entnehmen übersprungen (faule Löschung).
    Schlüsselvergleiche macht `heapq` in C, sie sind hier nicht zählbar (work = 0)."""

    def __init__(self, n=0, max_weight=None):
        super().__init__()
        self.heap, self.best = [], {}
        self.stale_pops = 0

    def __len__(self):
        return len(self.best)

    def push_or_decrease(self, node, key):
        if node in self.best:
            self.decrease_keys += 1
        else:
            self.pushes += 1
        self.best[node] = key
        heapq.heappush(self.heap, (key, node))

    def pop_min(self):
        while True:
            key, node = heapq.heappop(self.heap)
            if self.best.get(node) == key:
                del self.best[node]
                self.pops += 1
                return key, node
            self.stale_pops += 1


class DialQueue(_Counters):
    """Eimer nach ganzzahligem Schlüssel (Dial 1969): alle lebenden Schlüssel liegen in [aktuell, aktuell + größte Kantenkosten], also reichen C + 1 Eimer im Kreis.
    Nur für ganzzahlige, nichtnegative Kosten. work = Eimerschritte beim Suchen des nächsten nichtleeren Eimers."""

    def __init__(self, n=0, max_weight=1):
        super().__init__()
        self.size = int(max_weight) + 1
        self.buckets = [[] for _ in range(self.size)]
        self.key = {}
        self.cur = 0

    def __len__(self):
        return len(self.key)

    def push_or_decrease(self, node, key):
        if int(key) != key or key < 0:
            raise ValueError("Dial braucht ganzzahlige, nichtnegative Schlüssel")
        key = int(key)
        if node in self.key:
            self.decrease_keys += 1
        else:
            self.pushes += 1
        self.key[node] = key
        self.buckets[key % self.size].append((node, key))

    def pop_min(self):
        while True:
            bucket = self.buckets[self.cur % self.size]
            while bucket:
                node, key = bucket.pop()
                if self.key.get(node) == key:
                    del self.key[node]
                    self.pops += 1
                    return key, node
            self.cur += 1
            self.work += 1


class _FibNode:
    __slots__ = ("key", "node", "degree", "parent", "child", "left", "right", "mark")

    def __init__(self, key, node):
        self.key, self.node, self.degree, self.parent, self.child, self.mark = key, node, 0, None, None, False
        self.left = self.right = self


class FibonacciQueue(_Counters):
    """Fibonacci-Heap (Fredman und Tarjan 1984): Einfügen und Decrease-Key in O(1) amortisiert, Entnehmen O(log n) amortisiert - deshalb O(E + V log V) für Dijkstra."""

    def __init__(self, n=0, max_weight=None):
        super().__init__()
        self.min, self.nodes, self.count = None, {}, 0

    def __len__(self):
        return self.count

    @staticmethod
    def _splice(a, b):
        """Fügt die kreisförmige Liste b hinter a ein (beide nichtleer)."""
        a_right, b_left = a.right, b.left
        a.right, b.left = b, a
        b_left.right, a_right.left = a_right, b_left

    def _add_root(self, x):
        x.parent, x.left, x.right = None, x, x
        if self.min is None:
            self.min = x
        else:
            self._splice(self.min, x)
            self.work += 1
            if x.key < self.min.key:
                self.min = x

    def push_or_decrease(self, node, key):
        x = self.nodes.get(node)
        if x is None:
            self.pushes += 1
            x = self.nodes[node] = _FibNode(key, node)
            self._add_root(x)
            self.count += 1
            return
        self.decrease_keys += 1
        x.key = key
        p = x.parent
        if p is not None:
            self.work += 1
            if x.key < p.key:
                self._cut(x, p)
                self._cascade(p)
        self.work += 1
        if x.key < self.min.key:
            self.min = x

    def _unlink(self, x):
        if x.right is x:
            return None
        x.left.right, x.right.left = x.right, x.left
        return x.right

    def _cut(self, x, p):
        if p.child is x:
            p.child = self._unlink(x)
        else:
            self._unlink(x)
        p.degree -= 1
        x.mark = False
        self._add_root(x)

    def _cascade(self, y):
        while y.parent is not None:
            if not y.mark:
                y.mark = True
                return
            p = y.parent
            self._cut(y, p)
            y = p

    def pop_min(self):
        z = self.min
        if z.child is not None:
            children = []
            c = z.child
            while True:
                children.append(c)
                c = c.right
                if c is z.child:
                    break
            for c in children:
                c.parent = None
        else:
            children = []
        roots = []
        r = z.right
        while r is not z:
            roots.append(r)
            r = r.right
        roots.extend(children)
        del self.nodes[z.node]
        self.count -= 1
        self.pops += 1
        if not roots:
            self.min = None
            return z.key, z.node
        table = {}
        for x in roots:
            x.left = x.right = x
            d = x.degree
            while d in table:
                y = table.pop(d)
                self.work += 1
                if y.key < x.key:
                    x, y = y, x
                y.parent = x                                                     # y wird Kind von x
                if x.child is None:
                    x.child, y.left, y.right = y, y, y
                else:
                    self._splice(x.child, y)
                x.degree += 1
                y.mark = False
                d += 1
            table[d] = x
        self.min = None
        for x in table.values():
            x.left = x.right = x
            if self.min is None:
                self.min = x
            else:
                self._splice(self.min, x)
                self.work += 1
                if x.key < self.min.key:
                    self.min = x
        return z.key, z.node


QUEUES = {"array": ArrayQueue, "binary": BinaryHeapQueue, "lazy": LazyHeapQueue, "dial": DialQueue, "fibonacci": FibonacciQueue}
QUEUE_LABELS = {"array": "Feld (lineare Suche)", "binary": "Binärheap (Decrease-Key)", "lazy": "Binärheap, faul (heapq)", "dial": "Dial-Eimer", "fibonacci": "Fibonacci-Heap"}
