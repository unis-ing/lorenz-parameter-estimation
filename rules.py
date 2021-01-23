import numpy as np


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
        elif cn == 1:
            get_pr = apply_thresholds_and_Tc
        elif cn == 2:
            get_pr = apply_Tc
        elif cn == 3:
            get_pr = apply_approx_thresholds_and_Tc

    return get_pr, rule_f

# ------------------------------------------------------------------
#                           rules
# ------------------------------------------------------------------
"""
Add new rules here. Must have function signature (s, p) and return a number.
"""

def no_update(s, p):
    return p.pr


def rule1(s, p):
    return (p.pr * (s.u - s.v) + p.mu * (s.u - s.x)) / (s.x - s.v)


def rule2z(s, p):
    its = int(p.Tc / p.dt)
    return -1 - np.mean(s.z_list[-its:])


def rule2w(s, p):
    its = int(p.Tc / p.dt)
    return -1 - np.mean(s.w_list[-its:])


# def rule0(s, p):
#     return (p.pr * (s.u - s.v)) / (s.x - s.v)

# def rule3(s, p):
#     return s.w - (s.u*s.v - p.mu*(s.w-s.z)) / p.B + p.pr


# ------------------------------------------------------------------
#                           conditions
# ------------------------------------------------------------------
"""
Add new conditions here. Must have function signature (s, p, rule_f, t) 
and return a number.
"""

def apply_nothing(s, p, rule_f, t):
    """ condition # 0 """

    return rule_f(s, p)


def apply_thresholds_and_Tc(s, p, rule_f, t):
    """ condition # 1 """

    if p.nudge == 'u':
        poserr = abs(s.x - s.u)
        velerr = abs(s.xt - s.ut)
    elif p.nudge == 'v':
        poserr = abs(s.y - s.v)
        velerr = abs(s.yt - s.vt)
    elif p.nudge == 'w':
        poserr = abs(s.z - s.w)
        velerr = abs(s.zt - s.wt)

    # check conditions are met
    c1 = p.T >= p.Tc
    c2 = poserr <= p.a
    c3 = velerr <= p.b
    c4 = (poserr > 0) & (velerr > 0)

    if c1 & c2 & c3 & c4:
        p.decrease_a()
        p.decrease_b()
        p.update_a_list()
        p.update_b_list()
        p.reset_T()

        return rule_f(s, p)

    else:
        p.update_T(t_curr=t, t_old=s.t_prev)
        return p.pr


def apply_Tc(s, p, rule_f, t):
    """ condition # 2 """

    if p.T >= p.Tc:
        p.reset_T()
        return rule_f(s, p)
        
    else:
        p.update_T(t_curr=t, t_old=s.t_prev)
        return p.pr

def apply_approx_thresholds_and_Tc(s, p, rule_f, t):
    """ condition # 3 """
    p.update_T(t_curr=t, t_old=s.t_prev)

    if s.t_prev <= p.dt: # need at least 2 elem's in x_list
        return p.pr
    else:
        if t != s.t_prev:
            poserr = abs(s.x - s.u)
            finite_diff = (s.x - s.x_) / (t - s.t_prev)
            velerr = abs(finite_diff - s.ut)

            # check conditions are met
            c1 = p.T >= p.Tc
            c2 = poserr <= p.a
            c3 = velerr <= p.b
            c4 = (poserr > 0) & (velerr > 0)

            if c1 & c2 & c3 & c4:
                p.decrease_a()
                p.decrease_b()
                p.update_a_list()
                p.update_b_list()
                p.reset_T()

                return rule_f(s, p)

    return p.pr
