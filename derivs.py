"""
	Functions implementing the time derivatives.
"""

def get_deriv_fs(nudge):
    """
    returns list of deriv functions based on nudge.
    """
    UT = ut
    VT = vt
    WT = wt

    if nudge == 'u':
        UT = ut_nudge
    elif nudge == 'v':
        VT = vt_nudge
    elif nudge == 'w':
        WT = wt_nudge

    return (XT, YT, ZT, UT, VT, WT)

def XT(s, p):
    return p.PR * (s.y - s.x)


def YT(s, p):
    return -p.PR * s.x - s.y - s.x * s.z


def ZT(s, p):
    return s.x * s.y - p.B*(s.z + p.PR + p.RA)


def ut(s, p):
    return p.pr * (s.v - s.u) - p.mu * (s.u - s.x)


def vt(s, p):
    return -p.pr * s.u - s.v - s.u * s.w


def wt(s, p):
    return s.u * s.v - p.B * (s.w + p.pr + p.RA)


def ut_nudge(s, p):  # nudge u with x
    return ut(s, p) - p.mu * (s.u - s.x)


def vt_nudge(s, p):  # nudge v with y
    return vt(s, p) - p.mu * (s.v - s.y)


def wt_nudge(s, p):  # nudge w with z
    return wt(s, p) - p.mu * (s.w - s.z)