import sys
from collections import defaultdict

def check_file(fname):
    lines = [l.strip() for l in open(fname).readlines()]
    lines = [l.split() for l in lines if l]
    counts = defaultdict(int)
    for i, l in enumerate(lines):
        true, pred, token = l
        if token.startswith('##'):
            counts[(lines[i-1][1], pred)] += 1
    print(counts)

if __name__ == '__main__':
    check_file(sys.argv[1])
