#!/usr/bin/env python3
"""Run each baseline in its isolated conda environment."""
import argparse, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = {"sd15":("icra2027-inpaint", ROOT/"sd_inpaint/infer.py"),
           "lama":("icra2027-lama", ROOT/"lama/infer.py"),
           "mat":("icra2027-mat", ROOT/"mat/infer.py")}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default=str(ROOT/"results"))
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS)); p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    for method in a.methods:
        env, script=METHODS[method]
        command=["conda","run","--no-capture-output","-n",env,"python",str(script),"--input",a.input,"--output",a.output]
        if a.overwrite: command.append("--overwrite")
        print("+", " ".join(command), flush=True); subprocess.run(command,check=True,cwd=ROOT)
if __name__=="__main__": main()

