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
        elif cn == 4:
            get_pr = test

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
        p.reset_T()

        return rule_f(s, p)

    else:
        te = s.tfe
        if len(te) > 0:
            p.update_T(t_curr=t, t_old=te[-1])
        return p.pr


def apply_Tc(s, p, rule_f, t):
    """ condition # 2 """

    if p.T >= p.Tc:
        p.reset_T()
        return rule_f(s, p)
        
    else:
        te = s.tfe
        if len(te) > 0:
            p.update_T(t_curr=t, t_old=te[-1])
        return p.pr


def apply_approx_thresholds_and_Tc(s, p, rule_f, t):
    """ condition # 3 """
    te = s.tfe

    if len(te) < 2: # need at least 2 elem's in x_list, tfe
        return p.pr
    else:
        p.update_T(t_curr=t, t_old=te[-1])
        t1 = te[-2]
        t2 = te[-1]
        if t1 != t2: # sometimes nudged_lorenz is eval'd at same t twice
            x1 = s.x_list[-2]
            x2 = s.x_list[-1]
            poserr = abs(x2 - x1)
            xt_approx = (x2 - x1) / (t2 - t1)
            velerr = abs(xt_approx - s.ut)

            # check conditions are met
            c1 = p.T >= p.Tc
            c2 = poserr <= p.a
            c3 = velerr <= p.b

            if c1 & c2 & c3:
                p.decrease_a()
                p.decrease_b()
                p.reset_T()

                return rule_f(s, p)

    return p.pr


#----------------------------------------------
def test(s, p, rule_f, t):
    """ condition # 4 """
    te = s.tfe

    if len(te) < 2: # need at least 2 elem's in x_list, tfe
        return p.pr
    else:
        p.update_T(t_curr=t, t_old=te[-1])
        t1 = te[-2]
        t2 = te[-1]
        if t1 != t2: # sometimes nudged_lorenz is eval'd at same t twice
            x1 = s.x_list[-2]
            x2 = s.x_list[-1]
            poserr = abs(x2 - x1)
            xt_approx = (x2 - x1) / (t2 - t1)
            velerr = abs(xt_approx - s.ut)

            # check conditions are met
            c1 = p.T >= p.Tc
            # c2 = poserr <= p.a
            # c3 = velerr <= p.b

            if c1:
                p.reset_T()
                # if te[-1] >= 30:
                #     p.a *= 0.1

                return rule_f(s, p)

    return p.pr
