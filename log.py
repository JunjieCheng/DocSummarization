import math
import time


def as_hours(s):
    h = math.floor(s / 3600)
    m = math.floor(s / 60) - h * 60
    s -= m * 60 + 3600 * h
    return '%dh %dm %ds' % (h, m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_hours(s), as_hours(rs))
