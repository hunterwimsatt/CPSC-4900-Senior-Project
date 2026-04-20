from z3 import *

def run_smt_code_for_sat(smt_code):
    s = Solver()
    try:
        s.from_string(smt_code)
        return s.check()
    except:
        return "error"