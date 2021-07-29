'''utilities to build progress indicators

math functions:
    differentiate: takes derivatives
    ewma: computes an exponentially-weighted moving average
    clip: clips a number to [0, 1]
terminal utilities:
    terminal_size: returns the height and width of the terminal
    terminal_height: returns the height of the terminal
    terminal_width: returns the width of the terminal
timekeeping utilities:
    timestamp: returns a timestamp; same as time.time()
    TimeEstimator: extrapolates how long something will take
progress indicators:
    percent: prints a value in percent
    block_progressbar: draws a progressbar with Unicode block characters
    format_timespan: displays a number of seconds in an 'H:MM:SS' format
    with_annotations: adds annotations around a progressbar
    with_brackets: adds brackets around a progressbar
    with_eta: adds a time estimate around a progressbar
    with_percent: adds a percent readout around a progressbar
'''

import numpy as np
from functools import wraps
from struct import Struct


def differentiate(x, y):
    '''computes an estimate of the derivate dy/dx

    This is a dead-simple finite-difference scheme; should be good enough for
    a time estimate.

    Arguments:
    x: independent variable, assumed monotonically increasing, 1-D
    y: dependent variable, same shape as x

    Returns:
    (v, w), where v are the center points between values of x where the
    estimate should be best, and w is the derivative

    Examples:
    >>> differentiate([0, 1], [0, 1])
    (array([0.5]), array([1.]))
    >>> differentiate([0], [0])
    (array([0.]), array([nan]))
    >>> differentiate([0, 2, 3], [0, 1, 2])
    (array([1. , 2.5]), array([0.5, 1. ]))
    '''
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError('inputs must have same shape')
    if len(x) <= 1:
        return x, np.full_like(x, np.nan)
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return x[:-1] + dx / 2, dy / dx


def ewma(x, y, base=10, max_age=None):
    '''exponentially-weighted moving average

    Arguments:
    x, y: samples, x is assumed monotonically increasing
    base: base of the exponent
           - 1: uniform scaling
           - 1 + small_number: near-linear-scaling,
           - default: 10
    max_age: if passed, ignore samples older than this
              - default is None, which isn't really a moving average,
                but I wasn't sure what to set it to
    Returns:
    EMWA of y

    x is shifted and scaled s.t. x[0] = 0 and x[1] = 1
    weights are computed as w = base**x; w /= sum(w)

    Examples:
    >>> ewma([0, 1, 2, 3, 4, 5], [ 0, 0, 0, 10, 10, 10 ], base=1)
    5.0
    >>> ewma([0, 1, 2, 3, 4, 5], [ 0, 0, 0, 10, 10, 10 ], base=10)
    7.992399910868981
    >>> ewma([0, 1, 2, 3, 4, 5], [ 0, 0, 0, 10, 10, 10 ], base=100)
    9.406490568972323
    >>> ewma([0, 4, 8], [0, 10, 20], base=1, max_age=None)
    10.0
    >>> ewma([0, 4, 8], [0, 10, 20], base=1, max_age=8)
    10.0
    >>> ewma([0, 4, 8], [0, 10, 20], base=1, max_age=4)
    15.0
    >>> ewma([0, 4, 8], [0, 10, 20], base=1, max_age=0)
    20.0
    '''
    if not y:
        return np.nan
    if max_age is not None:
        if max_age < 0:
            raise ValueError('max_age must be positive')
        i = next(i for i in range(len(x)) if x[-1] - x[i] <= max_age)
        x, y = x[i:], y[i:]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError('inputs must have same shape')
    if len(x) == 1:
        return y[0]
    x -= x[0]
    x /= x[-1]
    w = base**x
    w /= sum(w)
    return w @ y


def clip(x, min=0, max=1):
    '''clips x to some bounds (sort of like numpy.clip)'''
    if x < min:
        return min
    if x > max:
        return max
    return x


def terminal_size(full=False):
    '''return the terminal size as reported by ioctl

    full: if True, return all four fields of the winsize structure; the last
          two are supposed to hold the size in pixels, but they're usually
          just zeros
    '''
    from fcntl import ioctl
    from termios import TIOCGWINSZ
    buf = bytearray(8)
    assert 0 == ioctl(1, TIOCGWINSZ, buf)
    return terminal_size.struct.unpack(buf)[:None if full else 2]
terminal_size.struct = Struct('HHHH')  # noqa: E305


def terminal_height():
    return terminal_size()[0]


def terminal_width():
    return terminal_size()[1]


def timestamp():
    from time import time
    return time()


def percent(progress, decimals=0, symbol=' %'):
    '''prints a percent; slightly more flexible than '{:%}'.format

    >>> percent(0.2)
    '20 %'
    >>> percent(0.2, 2)
    '20.00 %'
    >>> percent(0.2, 2, '%')
    '20.00%'
    '''
    return f'{progress * 100:.{decimals}f}{symbol}'


class TimeEstimator:
    '''simple estimator of how long something will take to finish'''
    def __init__(self, progress=0.0, **ewma_kwargs):
        '''create the new estimator
        progress: initial progress (will be clipped to [0, 1], default: 0)
        '''
        progress = clip(progress)
        self.history = [ (timestamp(), progress) ]
        self.ewma_kwargs = ewma_kwargs

    def update(self, progress):
        '''update the time estimate with the new progress
        '''
        progress = clip(progress)
        now = timestamp()
        self.history.append((now, progress))
        # FIXME: recomputing the same derivatives is inefficient
        rate = ewma(*differentiate(*zip(*self.history)), **self.ewma_kwargs)
        return float('nan') if rate == 0.0 else (1 - progress) / rate


def block_progressbar(progress, width=None):
    r'''displays a progress bar of some width

    Examples:
    >>> block_progressbar(0, 10)
    '          '
    >>> block_progressbar(0.1, 10)
    '█         '
    >>> block_progressbar(0.5, 10)
    '█████     '
    >>> block_progressbar(0.9, 10)
    '█████████ '
    >>> block_progressbar(1.0, 10)
    '██████████'
    >>> ''.join(block_progressbar(i / 8, 1) for i in range(8))
    ' ▏▎▍▌▋▊▉'
    >>> block_progressbar(0.31415, 10)
    '███▏      '
    '''
    progress = clip(progress)
    if width is None:
        width = terminal_width()
    if width < block_progressbar.min_width:
        raise ValueError(f'{width=} must be at least {block_progressbar.min_width}')

    if progress == 1.0:
        return '█' * width

    x = progress * width
    i = int(x)
    f = int((x - i) * 8)
    return '█' * i + ' ▏▎▍▌▋▊▉'[f] + ' ' * (width - i - 1)
block_progressbar.min_width = 1  # noqa: E305


def format_timespan(span):
    '''display a number of seconds as hh:mm:ss, rounding up

    >>> format_timespan(0)
    '0:00:00'
    >>> format_timespan(0.1)
    '0:00:01'
    >>> format_timespan(1)
    '0:00:01'
    '''
    try:
        if np.isnan(span):
            raise ValueError
        h, s = divmod(span, 3600)
        m, s = divmod(s, 60)
        return f'{int(h)}:{int(m):02d}:{int(np.ceil(s)):02d}'
    except ValueError:
        return '??:??:??'


def with_annotations(progressbar, prefix='', suffix=''):
    n = len(prefix) + len(suffix)
    fmt = f'{prefix}{{}}{suffix}'.format

    @wraps(progressbar)
    def closure(progress, width=None):
        if width is None:
            width = terminal_width()
        if width < closure.min_width:
            raise ValueError(f'{width=} must be at least {block_progressbar.min_width}')
        return fmt(progressbar(progress, width - n))

    closure.min_width = n + progressbar.min_width
    return closure


def with_brackets(progressbar, brackets='••'):
    lb, rb = brackets
    return with_annotations(progressbar, lb, rb)


def with_percent(progressbar, percent=percent, percent_width=6, percent_left=False):
    if percent_left:
        fmt = f'{{percent:<{percent_width}}}{{bar}}'.format
    else:
        fmt = f'{{bar}}{{percent:>{percent_width}}}'.format

    @wraps(progressbar)
    def closure(progress, width=None):
        if width is None:
            width = terminal_width()
        if width < closure.min_width:
            raise ValueError(f'{width=} must be at least {block_progressbar.min_width}')
        bar = progressbar(progress, width - percent_width)
        prc = percent(progress)
        return fmt(bar=bar, percent=prc)

    closure.min_width = percent_width + progressbar.min_width
    return closure


def with_eta(progressbar, time_estimator=None, eta_width=9, eta_left=False, max_eta=1000 * 60 * 60 - 1):
    if eta_left:
        fmt = f'{{eta:<{eta_width}}}{{bar}}'.format
    else:
        fmt = f'{{bar}}{{eta:>{eta_width}}}'.format

    if time_estimator is None:
        time_estimator = TimeEstimator()

    @wraps(progressbar)
    def closure(progress, width=None):
        if width is None:
            width = terminal_width()
        if width < closure.min_width:
            raise ValueError(f'{width=} must be at least {block_progressbar.min_width}')
        bar = progressbar(progress, width - eta_width)
        eta = time_estimator.update(progress)
        return fmt(bar=bar, eta=format_timespan(eta))

    closure.min_width = eta_width + progressbar.min_width
    return closure


if __name__ == '__main__':
    from doctest import testmod

    testmod()

    #from time import sleep
    #progressbar = with_eta(with_annotations(with_percent(with_brackets(block_progressbar)), prefix='progress: '))

    #start = timestamp()
    #for i in range(1001):
    #    print(progressbar(i / 1000), end='\r')
    #    exp = start + i * 0.005
    #    act = timestamp()
    #    sleep(max(0, exp - act))
    #print()
