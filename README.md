
# gc

<p align="center">
	<img width="500" alt="Screenshot of network debugger GUI" src="https://github.com/user-attachments/assets/e6d4e96f-d313-4874-94ad-3622909efea4">
</p>

Objective-free, modular, feedback neural networks that learn via association.

(*Disclaimer: Very rough/early/speculative work*)

## Install and Run

**NOTE: Tested with Python 3.13**

Install `requirements.txt` using your favorite package manager.

Run the main network:
```
python main.py
```
This should open a debugger GUI window. Closing the debugger will stop and save the network.

To load and run a save:
```
python main.py --load "./saves/agt0"
```

Choose the size of the network:
```
python main.py --size 100
```

To view a saved network without running it:
```
python view_only.py "./saves/agt0"
```

Run the smaller testing network:
```
python simple_nets.py
```

## Using the Debugger GUI

- The grid of cells on the left hand side shows the modules within the network.
- Hold left click on a square in the grid to see:
	- Activations and weights, on the right side
	- Outgoing connections, as gray rectangles on the grid
- While holding left click, either hold shift or right click to select an outgoing connection or an activation layer, and see detailed information in the middle pane.

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

## References

- **Neuroscience: Exploring the Brain** (2016); M. F. Bear, B. W. Connors, M. A. Paradiso
- **A Path Towards Autonomous Machine Intelligence** (2022); Y. LeCun
- **Active Inference: The Free Energy Principle in Mind, Brain, and Behavior** (2022); T. Parr, G. Pezzulo, K. J. Friston
- **The Book of Why: The New Science of Cause and Effect** (2018); J. Pearl, D. Mackenzie
- **Conjectures and Refutations: The Growth of Scientific Knowledge** (1963); K. Popper
- **The Alberta Plan for AI Research** (2022); R. S. Sutton, M. Bowling, P. M. Pilarski

## Roadmap

Right now it kind of sucks, need to make it "actually be good":
- Learn useful representations
	- Of perceptual input, goals, and actions
	- Models of self, other agents, and the world
	- Language
	- Abstract ideas
- Interact intelligently with environments
	- Explore and learn
	- Plan and pursue goals
- Think and be creative 😬

## Methods and Goals

| (nothing necessarily against these, just not the goals of this project) |                                                                                                         |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ❌ Gradient descent                                                      | ✔️ Hebbian association                                                                                  |
| ❌ Learn by obeying training data                                        | ✔️ Learn by [conjectures and criticisms](https://www.thebritishacademy.ac.uk/documents/4924/46p039.pdf) |
| ❌ Reward                                                                | ✔️ Interact with environments **without** reward (😱)                                                   |
| ❌ Top-down control                                                      | ✔️ Bottom-up cooperation and competition                                                                |
| ❌ Goal is to predict data                                               | ✔️ Goal is to explain the world                                                                         |
| ❌ Training and inference                                                | ✔️ Life                                                                                                 |
| ❌ Fulfill specified objectives                                          | ✔️ Open ended improvement                                                                               |
| ❌ AI safety by subservience to humans                                   | ✔️ AI safety by individual freedom                                                                      |
| ❌ AIs that are tools                                                    | ✔️ AIs who are people                                                                                   |

---

**gc**
1. **g**eneral intelligence / **c**reativity
2. George Carlin
