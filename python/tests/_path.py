"""Puts python/ (the library) first on the import path for the scripts in this folder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
