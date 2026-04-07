import time

COLORS = ["\033[91m","\033[92m","\033[93m","\033[94m","\033[95m","\033[96m"]
RESET = "\033[0m"

class Element:
    def __init__(self, name, weight=1, k=1, t0=0,
                 start_pos=0, allowed_slots=None, set_name=""):
        self.name = name
        self.weight = weight
        self.k = k
        self.t0 = t0
        self.start_pos = start_pos
        self.allowed_slots = allowed_slots or []
        self.set_name = set_name
        self.local_index = self.allowed_slots.index(start_pos)

    def local_position(self, t):
        if t < self.t0:
            return self.start_pos
        moves = (t - self.t0) // self.k
        idx = (self.local_index + moves) % len(self.allowed_slots)
        return self.allowed_slots[idx]

    def __repr__(self):
        return (f"{self.name}(set={self.set_name}, weight={self.weight}, "
                f"k={self.k}, t0={self.t0}, start_pos={self.start_pos}, "
                f"allowed={self.allowed_slots})")


def print_initial(elements, num_slots, num_ticks):
    print("\n=== INITIAL PARAMETERS ===")
    print("Name | Set | Weight | k | t0 | StartPos | AllowedSlots")
    print("-" * 70)
    for e in elements:
        print(f"{e.name:4} | {e.set_name:3} | {e.weight:6} | {e.k:2} | {e.t0:2} | "
              f"{e.start_pos:8} | {e.allowed_slots}")
    print(f"\nnum_slots = {num_slots}")
    print(f"num_ticks = {num_ticks}")


def simulate(elements, num_slots, num_ticks, allow_overlap=True, global_shift=True, real_time=True):
    history = []

    for t in range(num_ticks):
        slots = [[] for _ in range(num_slots)]
        weights = [0] * num_slots

        shift = t if global_shift else 0

        for e in elements:
            local_pos = e.local_position(t)
            pos = (local_pos + shift) % num_slots

            # Overlap check only with other sets
            if not allow_overlap:
                if any(s for s in slots[pos] if s != e.name and 
                       any(el.name == s and el.set_name != e.set_name for el in elements)):
                    raise ValueError(f"Overlap forbidden: element {e.name} tried to occupy slot {pos} at t={t}")

            slots[pos].append(e.name)
            weights[pos] += e.weight

        history.append((slots, weights))

        if real_time:
            print(f"\nt={t} | shift={shift} | ", end="")
            for i in range(num_slots):
                if slots[i]:
                    s = "+".join(slots[i])
                    w = weights[i]
                    color = COLORS[i % len(COLORS)]
                    print(f"{color}{s}({w}){RESET} ", end="")
                else:
                    print("0 ", end="")
            print()
            time.sleep(1)
    return history


def print_table(history):
    print("\n=== TABLE ===")
    for t, (slots, weights) in enumerate(history):
        row = []
        for i in range(len(slots)):
            if slots[i]:
                row.append(f"{'+'.join(slots[i])}({weights[i]})")
            else:
                row.append("0")
        print(f"{t}: {row}")


# =========================
# SETUP
# =========================
num_slots = 6
num_ticks = 8

A_slots = [0, 4, 5]
B_slots = [2, 3]
D_slots = [1]

elements = [
    Element("A1", weight=1, k=2, start_pos=0, allowed_slots=A_slots, set_name="A"),
    Element("A2", weight=2, k=1, start_pos=4, allowed_slots=A_slots, set_name="A"),
    Element("A3", weight=3, k=3, start_pos=5, allowed_slots=A_slots, set_name="A"),

    Element("B1", weight=5, k=1, start_pos=2, allowed_slots=B_slots, set_name="B"),
    Element("B2", weight=7, k=2, start_pos=3, allowed_slots=B_slots, set_name="B"),

    Element("D1", weight=11, k=1, start_pos=1, allowed_slots=D_slots, set_name="D"),
]

allow_overlap = False   # forbid overlap between different sets
global_shift = True     # enable global shift

print_initial(elements, num_slots, num_ticks)
history = simulate(elements, num_slots, num_ticks, allow_overlap=allow_overlap, global_shift=global_shift)
print_table(history)