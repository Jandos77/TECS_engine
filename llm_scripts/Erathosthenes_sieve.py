# -*- coding: utf-8 -*-
import time
import random
from collections import defaultdict
from statistics import mean
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
# CONFIGURATION CLASS
# ─────────────────────────────────────────────────────────────
@dataclass
class SimulationConfig:
    num_slots: int = 100
    num_ticks: int = 100
    time_sleep: float = 0.05
    random_seed: int = 42
    allow_overlap: bool = True
    global_shift: bool = False
    real_time: bool = True
    slot_presets: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if self.num_ticks <= 0:
            raise ValueError("num_ticks must be > 0")
        if self.time_sleep < 0:
            raise ValueError("time_sleep cannot be negative")

        if not self.slot_presets:
            self.slot_presets = {
                "A": [i for i in range(self.num_slots) if i % 6 in [0, 4, 5]],
                "B": [i for i in range(self.num_slots) if i % 6 in [2, 3]],
                "D": [i for i in range(self.num_slots) if i % 6 == 1],
            }

        self.slot_presets["All"] = list(range(self.num_slots))


# ─────────────────────────────────────────────────────────────
# ELEMENT CLASS
# ─────────────────────────────────────────────────────────────
class Element:
    def __init__(self, name, weight=1, k=1, step=1, t0=0,
                 start_pos=0, allowed_slots=None, set_name="",
                 priority=0, stall_probability=0.0, aging_factor=0.5,
                 stop_mode=None, stop_value=None):

        self.name = name
        self.weight = weight
        self.k = k
        self.step = step
        self.t0 = t0
        self.start_pos = start_pos
        self.allowed_slots = allowed_slots or []
        self.set_name = set_name
        self.base_priority = priority
        self.priority = priority
        self.aging_factor = aging_factor
        self.stall_probability = stall_probability

        self.stop_mode = stop_mode
        self.stop_value = stop_value
        self.stopped = False

        self.local_index = (
            self.allowed_slots.index(start_pos)
            if start_pos in (allowed_slots or []) else 0
        )

        self.current_pos = start_pos
        self.last_pos = start_pos

        self.wait_time = 0
        self.total_wait = 0
        self.move_count = 0
        self.block_count = 0
        self.stall_count = 0
        self.stall_history = []

    def check_stop(self, pos):
        if not self.stop_mode:
            return False

        if self.stop_mode == "auto_last":
            return pos == self.allowed_slots[-1]

        if self.stop_mode == "first_hit":
            return pos == self.stop_value

        if self.stop_mode == "gte":
            return pos >= self.stop_value

        if self.stop_mode == "range":
            a, b = self.stop_value
            return a <= pos <= b

        if self.stop_mode == "custom":
            return self.stop_value(pos)

        return False

    def local_position(self, t):
        if self.stopped:
            return self.last_pos

        if t < self.t0:
            return self.last_pos

        moves = (t - self.t0) // self.k
        idx = (self.local_index + self.step * moves) % len(self.allowed_slots)
        pos = self.allowed_slots[idx]

        if self.check_stop(pos):
            self.stopped = True
            self.last_pos = pos
            return pos

        self.last_pos = pos
        return pos

    def update_dynamic_priority(self):
        self.priority = self.base_priority + self.wait_time * self.aging_factor
        return self.priority

    def should_stall(self):
        if self.stall_probability <= 0:
            return False
        return random.random() < self.stall_probability


# ─────────────────────────────────────────────────────────────
# METRICS CLASS
# ─────────────────────────────────────────────────────────────
class Metrics:
    def __init__(self, num_slots, num_ticks):
        self.num_slots = num_slots
        self.num_ticks = num_ticks
        self.collision_count = 0
        self.collision_prevented = 0
        self.stall_events = 0
        self.slot_usage = [0] * num_slots
        self.slot_weight_sum = [0] * num_slots
        self.element_stats = defaultdict(lambda: {
            "wait_times": [],
            "moves": 0,
            "blocks": 0,
            "stalls": 0,
            "priorities": []
        })
        self.history = []

    def record_slot_usage(self, slots, weights):
        for i in range(self.num_slots):
            if slots[i]:
                self.slot_usage[i] += 1
                self.slot_weight_sum[i] += weights[i]

    def record_collision(self, prevented=False):
        self.collision_count += 1
        if prevented:
            self.collision_prevented += 1

    def record_stall(self, element_name):
        self.stall_events += 1
        self.element_stats[element_name]["stalls"] += 1

    def record_element_stat(self, element):
        stats = self.element_stats[element.name]
        stats["wait_times"].append(element.wait_time)
        stats["moves"] = element.move_count
        stats["blocks"] = element.block_count
        stats["priorities"].append(element.priority)

    def get_summary(self):
        total = self.num_slots * self.num_ticks
        occupied = sum(self.slot_usage)
        return {
            "total_ticks": self.num_ticks,
            "slot_utilization": occupied / total * 100 if total > 0 else 0,
            "avg_slot_utilization": mean(self.slot_usage) / self.num_ticks * 100,
            "collision_rate": self.collision_count / self.num_ticks if self.num_ticks > 0 else 0,
            "prevention_rate": self.collision_prevented / self.collision_count * 100 if self.collision_count > 0 else 0,
            "stall_rate": self.stall_events / self.num_ticks if self.num_ticks > 0 else 0,
            "slot_usage_per_slot": [u / self.num_ticks * 100 for u in self.slot_usage],
            "avg_weight_per_slot": [w / max(1, self.slot_usage[i]) for i, w in enumerate(self.slot_weight_sum)],
            "element_avg_wait": {
                n: mean(s["wait_times"]) if s["wait_times"] else 0
                for n, s in self.element_stats.items()
            },
            "element_avg_priority": {
                n: mean(s["priorities"]) if s["priorities"] else 0
                for n, s in self.element_stats.items()
            }
        }


# ─────────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────────
COLORS = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
RESET = "\033[0m"

def print_initial(elements, config):
    print("\n=== INITIAL PARAMETERS ===")
    hdr = f"{'Name':<6}{'Set':<5}{'W':<3}{'k':<3}{'Step':<4}{'t0':<4}{'Start':<6}{'Pri':<5}{'StopMode':<10}{'StopVal':<10}Allowed"
    print(hdr)
    print("-" * len(hdr))
    for e in elements:
        print(f"{e.name:<6}{e.set_name:<5}{e.weight:<3}{e.k:<3}{e.step:<4}{e.t0:<4}"
              f"{e.start_pos:<6}{e.base_priority:<5}{str(e.stop_mode):<10}{str(e.stop_value):<10}{e.allowed_slots}")
    print(f"\nslots={config.num_slots} | ticks={config.num_ticks} | shift={config.global_shift} | overlap={config.allow_overlap}")


def print_table(history):
    print("\n=== STATE TABLE ===")
    for t, (slots, weights) in enumerate(history):
        row = []
        for i in range(len(slots)):
            if slots[i]:
                row.append(f"{'+'.join(slots[i])}({weights[i]})")
            else:
                row.append("0")
        print(f"{t:3d}: {row}")


def print_metrics(metrics, elements):
    summary = metrics.get_summary()
    print("\n" + "=" * 50)
    print("ANALYTICAL REPORT")
    print("=" * 50)
    print(f"Ticks: {summary['total_ticks']}")
    print(f"Utilization: {summary['slot_utilization']:.1f}% (total) / {summary['avg_slot_utilization']:.1f}% (avg)")
    print(f"Collisions: {metrics.collision_count} (blocked: {metrics.collision_prevented})")
    print(f"Stall events: {metrics.stall_events}")

    print("\nSlot load:")
    for i in range(metrics.num_slots):
        pct = summary["slot_usage_per_slot"][i]
        bar = "#" * int(pct / 2)
        print(f"  [{i}] {pct:5.1f}% | weight:{summary['avg_weight_per_slot'][i]:4.2f} {bar}")

    print("\nElements:")
    print(f"{'Name':<6}{'Set':<4}{'Moves':<5}{'Block':<5}{'Wait':<6}{'Prior':<6}{'Stall'}")
    for e in sorted(elements, key=lambda x: x.set_name + x.name):
        stats = metrics.element_stats[e.name]
        aw = summary["element_avg_wait"].get(e.name, 0)
        ap = summary["element_avg_priority"].get(e.name, e.base_priority)
        print(f"{e.name:<6}{e.set_name:<4}{stats['moves']:<5}{stats['blocks']:<5}"
              f"{aw:<6.2f}{ap:<6.2f}{'!'*stats['stalls']}")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────
def simulate(elements, config):
    if config.random_seed is not None:
        random.seed(config.random_seed)

    metrics = Metrics(config.num_slots, config.num_ticks)
    history = []

    for t in range(config.num_ticks):
        slots = [[] for _ in range(config.num_slots)]
        weights = [0] * config.num_slots
        slot_elements = [[] for _ in range(config.num_slots)]
        shift = t if config.global_shift else 0

        for e in elements:
            e.update_dynamic_priority()

        sorted_elements = sorted(elements, key=lambda e: (-e.priority, e.name))

        for e in sorted_elements:
            if t < e.t0:
                continue

            if e.should_stall():
                e.wait_time += 1
                metrics.record_stall(e.name)
                continue

            local_pos = e.local_position(t)
            pos = (local_pos + shift) % config.num_slots

            slots[pos].append(e.name)
            slot_elements[pos].append(e)
            weights[pos] += e.weight

            e.move_count += 1
            e.current_pos = pos
            e.last_pos = pos
            e.wait_time = 0

        metrics.record_slot_usage(slots, weights)
        history.append((slots, weights))

        if config.real_time:
            print(f"t={t:3d} | ", end="")
            for i in range(config.num_slots):
                if slots[i]:
                    print(f"{'+'.join(slots[i])}({weights[i]}) ", end="")
                else:
                    print("0 ", end="")
            print()
            time.sleep(config.time_sleep)

    for e in elements:
        metrics.record_element_stat(e)

    return history, metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Create config
    config = SimulationConfig(
        num_slots=100,
        num_ticks=100,
        time_sleep=0.01,
        random_seed=42,
        allow_overlap=True,
        global_shift=False,
        real_time=True,
    )

    elements = [
        Element("A1", k=2, step=2, start_pos=4, stop_mode="auto_last", allowed_slots=config.slot_presets["All"], 
                set_name="A", priority=10, stall_probability=0.0, aging_factor=0.8, weight=1),
        Element("A2", k=3, step=3, start_pos=6, stop_mode="auto_last", allowed_slots=config.slot_presets["All"], 
                set_name="A", priority=10, stall_probability=0.0, aging_factor=1.2, weight=1),
        Element("A3", k=5, step=5, start_pos=10, stop_mode="auto_last", allowed_slots=config.slot_presets["All"], 
                set_name="A", priority=10, stall_probability=0.0, aging_factor=0.5, weight=1),
        Element("B1", k=7, step=7, start_pos=14, stop_mode="auto_last", allowed_slots=config.slot_presets["All"], 
                set_name="B", priority=10, stall_probability=0.0, aging_factor=1.0, weight=1),
        Element("B2", k=11, step=11, start_pos=22, stop_mode="auto_last", allowed_slots=config.slot_presets["All"], 
                set_name="B", priority=10, stall_probability=0.0, aging_factor=0.3, weight=1),
        Element("D1", k=13, step=13, start_pos=26, stop_mode="auto_last", allowed_slots=config.slot_presets["All"], 
                set_name="D", priority=10, stall_probability=0.0, aging_factor=0.0, weight=1),
        Element("D2", k=17, step=17, start_pos=34, stop_mode="auto_last", allowed_slots=config.slot_presets["All"],
                set_name="D", priority=10, stall_probability=0.0, aging_factor=0.0, weight=1),
        Element("D3", k=19, step=19, start_pos=38, stop_mode="auto_last", allowed_slots=config.slot_presets["All"],
                set_name="D", priority=10, stall_probability=0.0, aging_factor=0.0, weight=1),
        Element("D4", k=23, step=23, start_pos=46, stop_mode="auto_last", allowed_slots=config.slot_presets["All"],
                set_name="D", priority=10, stall_probability=0.0, aging_factor=0.0, weight=1),
        Element("D5", k=29, step=29, start_pos=58, stop_mode="auto_last", allowed_slots=config.slot_presets["All"],
                set_name="D", priority=10, stall_probability=0.0, aging_factor=0.0, weight=1)   
    ]

    print_initial(elements, config)
    history, metrics = simulate(elements, config)
    print_table(history)
    print_metrics(metrics, elements)