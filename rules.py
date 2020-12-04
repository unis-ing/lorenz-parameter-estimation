import numpy as np

# ------------------------------------------------------------------
#							rules
# ------------------------------------------------------------------


def map_rule_to_f(rule):
    """
    returns get_pr function and rule_f corresponding to rule.

    rule should be a string formated as rule[#]c[#].
    """

    if rule == 'no_update':
        get_pr = apply_nothing
        rule_f = no_update

    else:
        i = rule.find('_c')
        cn = int(rule[i+2:]) # condition num
        rule_name = rule[:i]
        rule_f = globals()[rule_name]

        if cn == 0:
            get_pr = apply_nothing
        if cn == 1:
            get_pr = apply_thresholds_and_Tc
        elif cn == 2:
            get_pr = apply_Tc

    return get_pr, rule_f


def no_update(s, p):
    return p.pr


def rule1(s, p):
    return (p.pr * (s.u - s.v) + p.mu * (s.u - s.x)) / (s.x - s.v)


def rule2(s, p):
    its = int(p.Tc / p.dt)
    return -1 - np.mean(s.z_list[-its:])


def rule2a(s, p):
    its = int(p.Tc / p.dt)
    return -1 - np.mean(s.w_list[-its:])


def apply_nothing(s, p, rule_f):
    """ cn = 0 """
    return rule_f(s, p)


def apply_Tc(s, p, rule_f):
    """ cn = 2 """
    c = p.T >= p.Tc

    if c:
        p.reset_T()
        return rule_f(s, p)
    else:
        p.increase_T()
        return p.pr


def apply_thresholds_and_Tc(s, p, rule_f):
    """ cn = 1 """
    if p.nudge == 'u':
        poserr = abs(s.x - s.u)
        velerr = abs(s.xt - s.ut)
    elif p.nudge == 'v':
        poserr = abs(s.y - s.v)
        velerr = abs(s.yt - s.vt)
    elif p.nudge == 'w':
        poserr = abs(s.z - s.w)
        velerr = abs(s.zt - s.wt)

    c1 = p.T >= p.Tc
    c2 = poserr <= p.a
    c3 = velerr <= p.b
    c4 = poserr > 0 and velerr > 0
    c = c1 & c2 & c3 & c4

    if c:
        p.decrease_a()
        p.decrease_b()
        p.update_a_list()
        p.update_b_list()
        p.reset_T()

        return rule_f(s, p)

    else:
        p.increase_T()
        return p.pr