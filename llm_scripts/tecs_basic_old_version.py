import time

# Colors
COLORS = ["\033[91m","\033[92m","\033[93m","\033[94m","\033[95m","\033[96m"]
RESET = "\033[0m"


class Element:
    def __init__(self, name, weight=1, k=1, t0=0, start_pos=0, frozen=False, color_idx=0, set_name=""):
        self.name = name
        self.weight = weight
        self.k = k
        self.t0 = t0
        self.start_pos = start_pos
        self.frozen = frozen
        self.color_idx = color_idx
        self.set_name = set_name

    def current_position(self, t, num_slots):
        if self.frozen or t < self.t0:
            return self.start_pos
        moves = (t - self.t0) // self.k
        return (self.start_pos + moves) % num_slots

    def __repr__(self):
        return (f"{self.name}(set={self.set_name}, weight={self.weight}, "
                f"k={self.k}, t0={self.t0}, pos={self.start_pos})")


# ===== C as a set =====
def create_C_from_sets(sets):
    seen = set()
    C = []
    for elems in sets.values():
        for e in elems:
            if e.name not in seen:
                C.append(e)
                seen.add(e.name)
    return C


def print_initial_values(sets, C, num_slots, num_ticks):
    print("\n=== Initial element parameters ===")
    print("Set | Name | Weight | k | t0 | StartPos | Frozen")
    print("-" * 55)
    for set_name, elems in sets.items():
        for e in elems:
            print(f"{set_name:3} | {e.name:4} | {e.weight:6} | {e.k:2} | {e.t0:2} | {e.start_pos:8} | {e.frozen}")

    print("\n=== Set C ===")
    for i, e in enumerate(C):
        print(f"{i:2}: {e}")

    print(f"\nNumber of slots: {num_slots}")
    print(f"Number of ticks: {num_ticks}")


def simulate_slots(elements, num_slots, num_ticks, real_time=True):
    history = []

    for t in range(num_ticks):
        slots = [[] for _ in range(num_slots)]
        slot_weights = [0] * num_slots

        for e in elements:
            pos = e.current_position(t, num_slots)
            slots[pos].append(e.name)
            slot_weights[pos] += e.weight

        history.append((slots, slot_weights))

        if real_time:
            print(f"\nt={t} s | ", end="")
            for i in range(num_slots):
                if slots[i]:
                    content = "+".join(slots[i])
                    total = slot_weights[i]
                    color = COLORS[i % len(COLORS)]
                    print(f"{color}{content}({total}){RESET} ", end="")
                else:
                    print("0 ", end="")
            print()
            time.sleep(1)

    return history


def print_table(history):
    print("\n=== Slot table (names + sum of weights) ===")
    num_slots = len(history[0][0])

    print("Time | " + " | ".join([f"S{i}" for i in range(num_slots)]))
    print("-" * (7 + 12 * num_slots))

    for t, (slots, weights) in enumerate(history):
        row = []
        for i in range(num_slots):
            if slots[i]:
                row.append(f"{'+'.join(slots[i])}({weights[i]})")
            else:
                row.append("0")
        print(f"{t:3}  | " + " | ".join(row))


# =========================
# DATA
# =========================

A = [
    Element("A1", weight=1, k=2, t0=0, start_pos=0, set_name="A"),
    Element("A2", weight=1, k=1, t0=0, start_pos=1, set_name="A")
]

B = [
    Element("B1", weight=1, k=3, t0=0, start_pos=0, set_name="B"),
    Element("B2", weight=1, k=2, t0=1, start_pos=1, set_name="B"),
    Element("B3", weight=1, k=1, t0=0, start_pos=1, set_name="B")
]

D = [
    Element("D1", weight=1, k=1, t0=0, start_pos=2, set_name="D"),
]

sets = {"A": A, "D": D, "B": B}

num_slots = 2
num_ticks = 8

# ===== C =====
C = create_C_from_sets(sets)

# Output initial data
print_initial_values(sets, C, num_slots, num_ticks)

# Simulation
history = simulate_slots(C, num_slots, num_ticks, real_time=True)

# Table
print_table(history)