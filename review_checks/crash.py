import sys, core.photon_transport_adapters as ad, core.integration_runner as ir
from lw_integrator import cli
orig=ad.commit_general_transport; n=[0]
def wrapped(*a, **k):
    n[0]+=1
    if n[0]==75: raise RuntimeError("simulated crash")
    return orig(*a, **k)
ad.commit_general_transport=wrapped
try:
    cli.main(["--config", sys.argv[1], "--output", "/dev/null", "--quiet"])
except RuntimeError as e: print("crashed:", e)
