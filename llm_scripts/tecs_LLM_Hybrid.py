# -*- coding: utf-8 -*-
"""
Element-Slot + PyTorch GPU Simulation (CUDA-ready)
Full working version with corrected AMP usage
"""
import time
import random
from collections import defaultdict
from statistics import mean
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Console colors
COLORS = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
RESET = "\033[0m"

# --------------------- CONFIG CLASS ---------------------
@dataclass
class SimulationConfig:
    num_slots: int = 6
    num_ticks: int = 15
    time_sleep: float = 0.05
    random_seed: int = 42
    allow_overlap: bool = False
    global_shift: bool = True
    real_time: bool = True
    device: str = "cuda"

# --------------------- ELEMENT CLASS ---------------------
class Element:
    def __init__(self, name, weight=1, k=1, t0=0,
                 start_pos=0, allowed_slots=None, set_name="",
                 priority=0, stall_probability=0.0, aging_factor=0.5):
        self.name = name
        self.weight = weight
        self.k = k
        self.t0 = t0
        self.start_pos = start_pos
        self.allowed_slots = allowed_slots or []
        self.set_name = set_name
        self.base_priority = priority
        self.priority = priority
        self.aging_factor = aging_factor
        self.stall_probability = stall_probability
        self.local_index = self.allowed_slots.index(start_pos) if start_pos in (allowed_slots or []) else 0
        self.current_pos = start_pos
        self.wait_time = 0
        self.total_wait = 0
        self.move_count = 0
        self.block_count = 0
        self.stall_count = 0
        self.stall_history = []

    def local_position(self, t):
        if t < self.t0:
            return self.start_pos
        moves = (t - self.t0) // self.k
        idx = (self.local_index + moves) % len(self.allowed_slots)
        return self.allowed_slots[idx]

    def update_dynamic_priority(self):
        self.priority = self.base_priority + self.wait_time * self.aging_factor
        return self.priority

    def should_stall(self):
        return random.random() < self.stall_probability if self.stall_probability > 0 else False

    def __repr__(self):
        return f"{self.name}(set={self.set_name}, w={self.weight}, k={self.k}, pri={self.base_priority})"

# --------------------- METRICS CLASS ---------------------
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

    def record_slot_usage(self, slots, weights, t):
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
            "element_avg_wait": {n: mean(s["wait_times"]) if s["wait_times"] else 0 for n, s in self.element_stats.items()},
            "element_avg_priority": {n: mean(s["priorities"]) if s["priorities"] else 0 for n, s in self.element_stats.items()}
        }

# --------------------- PRINT FUNCTIONS ---------------------
def print_initial(elements, num_slots, num_ticks):
    print("\n=== INITIAL PARAMETERS === ")
    hdr = f"{'Name': <6}{'Set': <5}{'W': <3}{'k': <3}{'t0': <3}{'Start': <6}{'Pri': <5}{'Age': <5}{'Stall%': <6}Allowed "
    print(hdr)
    print("-" * len(hdr))
    for e in elements:
        print(f"{e.name: <6}{e.set_name: <5}{e.weight: <3}{e.k: <3}{e.t0: <3} "
              f"{e.start_pos: <6}{e.base_priority: <5}{e.aging_factor: <5.2f} "
              f"{e.stall_probability*100: <6.1f}%{e.allowed_slots} ")
    print(f"\nslots={num_slots} | ticks={num_ticks} | shift=True | overlap=False ")

def print_table(history):
    print("\n=== STATE TABLE ===")
    for t, (slots, weights) in enumerate(history):
        row = []
        for i in range(len(slots)):
            if slots[i]:
                row.append(f"{'+'.join(slots[i])}({weights[i]})")
            else:
                row.append("0")
        print(f"{t:2d}: {row}")

def print_metrics(metrics, elements):
    summary = metrics.get_summary()
    print("\n" + "=" * 50)
    print("ANALYTICAL REPORT ")
    print("=" * 50)
    print(f"Ticks: {summary['total_ticks']} ")
    print(f"Utilization: {summary['slot_utilization']:.1f}% (total) / {summary['avg_slot_utilization']:.1f}% (avg) ")
    print(f"Collisions: {metrics.collision_count} (blocked: {metrics.collision_prevented}) ")
    print(f"Stalls: {metrics.stall_events} ")
    print("\nSlot load: ")
    for i in range(metrics.num_slots):
        pct = summary["slot_usage_per_slot"][i]
        bar = "#" * int(pct / 5)
        print(f"  [{i}] {pct:5.1f}% | weight:{summary['avg_weight_per_slot'][i]:4.2f} {bar} ")
        
    print("\nElements: ")
    print(f"{'Name': <6}{'Set': <4}{'Moves': <5}{'Blocks': <5}{'Wait': <6}{'Priority': <6}{'Stall'} ")
    for e in sorted(elements, key=lambda x: x.set_name + x.name):
        stats = metrics.element_stats[e.name]
        aw = summary["element_avg_wait"].get(e.name, 0)
        ap = summary["element_avg_priority"].get(e.name, e.base_priority)
        print(f"{e.name: <6}{e.set_name: <4}{stats['moves']: <5}{stats['blocks']: <5}{aw: <6.2f}{ap: <6.2f}{'!'*stats['stalls']} ")
    print("=" * 50)

# --------------------- SIMULATION ---------------------
def simulate(elements, config: SimulationConfig):
    if config.random_seed is not None:
        random.seed(config.random_seed)
        
    metrics = Metrics(config.num_slots, config.num_ticks)
    history = []
    elem_by_name = {e.name: e for e in elements}

    for t in range(config.num_ticks):
        slots = [[] for _ in range(config.num_slots)]
        slot_elements = [[] for _ in range(config.num_slots)]
        weights = [0] * config.num_slots
        shift = t if config.global_shift else 0

        for e in elements:
            e.update_dynamic_priority()

        sorted_elements = sorted(elements, key=lambda e: (-e.priority, e.name))

        for e in sorted_elements:
            if t < e.t0:
                continue

            if e.should_stall():
                e.stall_count += 1
                e.stall_history.append(t)
                e.wait_time += 1
                e.total_wait += 1
                metrics.record_stall(e.name)
                metrics.element_stats[e.name]["wait_times"].append(e.wait_time)
                continue

            local_pos = e.local_position(t)
            pos = (local_pos + shift) % config.num_slots

            if not config.allow_overlap:
                conflict = any(existing.set_name != e.set_name for existing in slot_elements[pos])
                if conflict:
                    metrics.record_collision(prevented=True)
                    e.block_count += 1
                    e.wait_time += 1
                    e.total_wait += 1
                    metrics.element_stats[e.name]["wait_times"].append(e.wait_time)
                    continue

            # Place element
            slots[pos].append(e.name)
            slot_elements[pos].append(e)
            weights[pos] += e.weight
            e.move_count += 1
            e.current_pos = pos
            e.wait_time = 0
            metrics.element_stats[e.name]["moves"] = e.move_count
            metrics.element_stats[e.name]["blocks"] = e.block_count

        metrics.record_slot_usage(slots, weights, t)
        history.append((slots, weights))

        if config.real_time:
            print(f"\nt={t:2d} | shift={shift:2d} |  ", end=" ")
            for i in range(config.num_slots):
                if slots[i]:
                    s = "+ ".join(slots[i])
                    w = weights[i]
                    marker = " ".join("^" for ename in slots[i] if elem_by_name[ename].priority > elem_by_name[ename].base_priority + 2)
                    color = COLORS[i % len(COLORS)]
                    print(f"{color}{s}{marker}({w}){RESET}  ", end=" ")
                else:
                    print("0  ", end=" ")
            stalled = [e.name for e in elements if t in e.stall_history]
            if stalled:
                print(f" | STALL:{','.join(stalled)} ", end=" ")
            print()
            time.sleep(config.time_sleep)

    for e in elements:
        metrics.record_element_stat(e)
    metrics.history = history
    return history, metrics

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    config = SimulationConfig(
        num_slots=6,
        num_ticks=15,
        time_sleep=0.05,
        random_seed=42,
        allow_overlap=False,
        global_shift=True,
        real_time=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"🚀 Device: {config.device}")

    A_slots = [0, 4, 5]
    B_slots = [2, 3]
    D_slots = [1]

    elements = [
        Element("A1", k=2, start_pos=0, allowed_slots=A_slots, set_name="A", priority=10, stall_probability=0.1, aging_factor=0.8),
        Element("A2", k=1, start_pos=4, allowed_slots=A_slots, set_name="A", priority=5, stall_probability=0.2, aging_factor=1.2),
        Element("A3", k=3, start_pos=5, allowed_slots=A_slots, set_name="A", priority=8, stall_probability=0.05, aging_factor=0.5),
        Element("B1", k=1, start_pos=2, allowed_slots=B_slots, set_name="B", priority=7, stall_probability=0.15, aging_factor=1.0),
        Element("B2", k=2, start_pos=3, allowed_slots=B_slots, set_name="B", priority=3, stall_probability=0.0, aging_factor=0.3),
        Element("D1", k=1, start_pos=1, allowed_slots=D_slots, set_name="D", priority=15, stall_probability=0.0, aging_factor=0.0, weight=2)
    ]

    print_initial(elements, config.num_slots, config.num_ticks)
    print(f"Seed: {config.random_seed}\n ")

    history, metrics = simulate(elements, config)

    print_table(history)
    print_metrics(metrics, elements)