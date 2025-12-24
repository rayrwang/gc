
# gc

Objective-free, modular, feedback neural networks that learn via association.

(*Disclaimer: Very rough/early/speculative work*)

## Install and Run

Install requirements:
```
# Linux
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run the main network:
```
python main.py
```
This should open a debugger GUI window.

## Files

```
src/
|-- funcs.py     # Functions used in agents
|-- agents.py    # Main neural nets file
|-- iotypes.py   # Input and output types
|-- envs.py      # Virtual environments or software interfaces to the real world
|-- debugger.py  # Debugger GUI
main.py

simple/          # Snaller neural nets for testing
|-- funcs.py
|-- agents.py
|-- debugger.py
simple_nets.py
```
