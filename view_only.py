
import sys

from src.agents import Agt

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Expected 1 command line argument, got {len(sys.argv)-1}!"
    server = Agt.load(sys.argv[1])
    server.debug_init()
    while True:
        server.debug_update()
        if server.pipes["overview"][0].poll():
            sys.exit()
