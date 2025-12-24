
import time
from itertools import count

from tqdm import tqdm

from simple.agents import Ising, Oscillator, NrnAgt

def ising():
    agt = Ising("./saves/ising", 2000)
    agt.debug_init()
    for _ in tqdm(count()):
        agt.step()

def oscillator():
    agt = Oscillator("./saves/oscillator")
    agt.debug_init()
    for _ in tqdm(count()):
        agt.step()

def nrns():
    agt = NrnAgt("./saves/nrns", 2000)
    agt.debug_init()
    for _ in tqdm(count()):
        agt.step()

if __name__ == "__main__":
    # ising()
    # oscillator()
    nrns()
