import random
import re
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        # trigram counts
        self.tri = defaultdict(Counter)
        # bigram counts
        self.bi = defaultdict(Counter)
        # unigram counts
        self.uni = Counter()

        # word pattern
        self.pat = re.compile(r"[A-Za-z0-9']+")

        # start and end tokens
        self.s = "<s>"
        self.e = "</s>"

    def _tok(self, txt):
        # clean and split
        return self.pat.findall(txt.lower())

    def _split_sent(self, txt):
        # break into sentences
        raw = re.split(r"[.!?]+", txt)
        return [r.strip() for r in raw if r.strip()]

    def fit(self, text):
        # skip empty input
        if not text:
            return

        sents = self._split_sent(text)

        for s in sents:
            w = self._tok(s)

            # update unigram
            for x in w:
                self.uni[x] += 1
            self.uni[self.e] += 1

            # pad for trigram window
            pad = [self.s, self.s] + w + [self.e]

            for i in range(len(pad) - 2):
                a, b, c = pad[i], pad[i+1], pad[i+2]
                self.tri[(a, b)][c] += 1
                self.bi[b][c] += 1

    def _pick(self, box):
        # probabilistic choice
        if not box:
            return None
        k = list(box.keys())
        v = list(box.values())
        return random.choices(k, v, k=1)[0]

    def generate(self, max_length=50):
        # start context
        a, b = self.s, self.s
        out = []

        if not self.uni:
            return ""

        for _ in range(max_length):
            nxt = self._pick(self.tri.get((a, b), {}))  # trigram
            if nxt is None:
                nxt = self._pick(self.bi.get(b, {}))     # bigram
            if nxt is None:
                nxt = self._pick(self.uni)              # unigram

            if nxt is None or nxt == self.e:
                break

            out.append(nxt)
            a, b = b, nxt

        return " ".join(out)
