# source code for multilingual project on unified translation of multilingual NL and PL.
import os
import sys

src_dir = os.path.abspath("src")
sys.path.append(src_dir)
evaluator_dir = os.path.join(src_dir, "evaluator")
sys.path.append(evaluator_dir)