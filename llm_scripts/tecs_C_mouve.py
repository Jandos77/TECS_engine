# -*- coding: utf-8 -*-
import time
import random

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
        # [!] FIX #3: protection against ValueError if start_pos is not in allowed_slots
        self.local_index = (self.allowed_slots.index(start_pos) 
                           if start_pos in self.allowed_slots else 0)

    def local_position(self, t):
        if t < self.t0:
            return self.start_pos
        moves = (t - self.t0) // self.k
        idx = (self.local_index + moves) % len(self.allowed_slots)
        return self.allowed_slots[idx]

    def __repr__(self):
        return f"{self.name}(set={self.set_name}, k={self.k})"


def print_initial(elements, num_slots, num_ticks, time_sleep):
    print("\n=== INITIAL PARAMETERS ===")
    print("Name | Set | Weight | k | t0 | StartPos | AllowedSlots")
    print("-" * 70)
    for e in elements:
        print(f"{e.name:4} | {e.set_name:3} | {e.weight:6} | {e.k:2} | {e.t0:2} | "
              f"{e.start_pos:8} | {e.allowed_slots}")
    print(f"\nnum_slots = {num_slots}")
    print(f"num_ticks = {num_ticks}")
    print(f"time_sleep = {time_sleep}")


# [!] FIX #1: added time_sleep parameter to function signature
def simulate(elements, num_slots, num_ticks, allow_overlap=True, 
             global_shift=True, real_time=True, time_sleep=0.3):
    history = []

    for t in range(num_ticks):
        # [!] FIX #2: store Element objects instead of names — simplifies collision check
        slots = [[] for _ in range(num_slots)]
        weights = [0] * num_slots
        shift = t if global_shift else 0

        for e in elements:
            if t < e.t0:
                continue
                
            local_pos = e.local_position(t)
            pos = (local_pos + shift) % num_slots

            # Collision check between different sets
            if not allow_overlap:
                conflict = False
                for existing in slots[pos]:
                    if existing.set_name != e.set_name:
                        conflict = True
                        break
                
                if conflict:
                    # Skip placement instead of raise — simulation continues
                    continue

            slots[pos].append(e)  # [v] store object
            weights[pos] += e.weight

        # For history, convert objects back to names (for compatibility with print_table)
        history.append((
            [[el.name for el in slot] for slot in slots],
            weights
        ))

        if real_time:
            print(f"\nt={t} | shift={shift} | ", end="")
            for i in range(num_slots):
                if slots[i]:
                    s = "+".join(el.name for el in slots[i])
                    w = weights[i]
                    color = COLORS[i % len(COLORS)]
                    print(f"{color}{s}({w}){RESET} ", end="")
                else:
                    print("0 ", end="")
            print()
            time.sleep(time_sleep)  # [v] time_sleep is now in scope
            
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


if __name__ == "__main__":
    num_slots = 6
    num_ticks = 8
    time_sleep = 0.3  # parameter is now correctly passed to simulate()

    A_slots = [0, 4, 5]
    B_slots = [2, 3]
    D_slots = [1]

    elements = [
        Element("A1", k=2, start_pos=0, allowed_slots=A_slots, set_name="A"),
        Element("A2", k=1, start_pos=4, allowed_slots=A_slots, set_name="A"),
        Element("A3", k=3, start_pos=5, allowed_slots=A_slots, set_name="A"),
        Element("B1", k=1, start_pos=2, allowed_slots=B_slots, set_name="B"),
        Element("B2", k=2, start_pos=3, allowed_slots=B_slots, set_name="B"),
        Element("D1", k=1, start_pos=1, allowed_slots=D_slots, set_name="D"),
    ]

    allow_overlap = False
    global_shift = True

    print_initial(elements, num_slots, num_ticks, time_sleep)
    
    history = simulate(
        elements, num_slots, num_ticks,
        allow_overlap=allow_overlap,
        global_shift=global_shift,
        real_time=True,
        time_sleep=time_sleep  # [v] pass explicitly
    )
    
    print_table(history)