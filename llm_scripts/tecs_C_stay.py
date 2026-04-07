import time
from itertools import cycle

# Colors for sets
COLORS = ["\033[91m","\033[92m","\033[93m","\033[94m","\033[95m","\033[96m"]
RESET = "\033[0m"

class Element:
    def __init__(self, name, weight=1, k=1, t0=0, start_pos=0, frozen=False, color_idx=0, set_name="", zone=None):
        self.name = name
        self.weight = weight
        self.k = k
        self.t0 = t0
        self.start_pos = start_pos
        self.frozen = frozen
        self.color_idx = color_idx
        self.set_name = set_name
        self.zone = zone  # allowed slots

    def current_position(self, t):
        if self.frozen or t < self.t0:
            return self.zone[self.start_pos]
        moves = (t - self.t0) // self.k
        idx = (self.start_pos + moves) % len(self.zone)
        return self.zone[idx]

def print_initial_values(sets):
    print("=== Initial element parameters ===")
    print("Set | Name | Weight | k | t0 | StartPos | Frozen | Zone")
    print("-" * 70)
    for set_name, elems in sets.items():
        for e in elems:
            print(f"{set_name:3} | {e.name:4} | {e.weight:6} | {e.k:1} | {e.t0:2} | {e.start_pos:8} | {e.frozen} | {e.zone}")

def animate_slots(sets, num_slots, num_ticks, delay=0.1):
    all_elements = [e for elems in sets.values() for e in elems]
    print("\n=== Element movement animation ===")
    for t in range(num_ticks):
        slots = [0] * num_slots
        slot_names = [""] * num_slots
        slot_colors = [None] * num_slots
        for e in all_elements:
            pos = e.current_position(t)
            slots[pos] += e.weight
            slot_colors[pos] = e.color_idx
            if slot_names[pos]:
                slot_names[pos] += f"+{e.name}"
            else:
                slot_names[pos] = e.name
        # Output current tick
        print(f"\nt={t} s | ", end="")
        for i in range(num_slots):
            val = f"{slot_names[i]}({slots[i]})" if slot_names[i] else "0"
            color = COLORS[slot_colors[i] % len(COLORS)] if slot_colors[i] is not None else ""
            print(f"{color}{val}{RESET} ", end="")
        print()
        time.sleep(delay)

# =========================
# Setup Example
# =========================

num_slots = 6
num_ticks = 10

time_sleep = 0.3

# Sets and zones
A = [
    Element("A1", weight=1, k=2, t0=0, start_pos=0, color_idx=0, set_name="A", zone=[0,4,5]),
    Element("A2", weight=1, k=1, t0=0, start_pos=1, color_idx=0, set_name="A", zone=[0,4,5]),
    Element("A3", weight=1, k=3, t0=0, start_pos=2, color_idx=0, set_name="A", zone=[0,4,5])
]

B = [
    Element("B1", weight=1, k=2, t0=0, start_pos=0, color_idx=1, set_name="B", zone=[2,3]),
    Element("B2", weight=1, k=1, t0=0, start_pos=1, color_idx=1, set_name="B", zone=[2,3])
]

D = [
    Element("D1", weight=1, k=1, t0=0, start_pos=0, color_idx=2, set_name="D", zone=[1])
]

sets = {"A": A, "B": B, "D": D}

# Initial values
print_initial_values(sets)
print(f"\nNumber of slots: {num_slots}")
print(f"Number of ticks: {num_ticks}")

# Movement animation
animate_slots(sets, num_slots, num_ticks, delay=time_sleep)