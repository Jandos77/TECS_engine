import time
from collections import defaultdict
from statistics import mean

COLORS = ["\033[91m","\033[92m","\033[93m","\033[94m","\033[95m","\033[96m"]
RESET = "\033[0m"

class Element:
    def __init__(self, name, weight=1, k=1, t0=0,
                 start_pos=0, allowed_slots=None, set_name="", priority=0):
        self.name = name
        self.weight = weight
        self.k = k
        self.t0 = t0
        self.start_pos = start_pos
        self.allowed_slots = allowed_slots or []
        self.set_name = set_name
        self.priority = priority  # priority in conflicts (higher = more important)
        self.local_index = self.allowed_slots.index(start_pos) if start_pos in (allowed_slots or []) else 0
        # Element Statistics
        self.wait_time = 0  # total wait time
        self.move_count = 0  # number of successful moves
        self.block_count = 0  # number of blocks

    def local_position(self, t):
        if t < self.t0:
            return self.start_pos
        moves = (t - self.t0) // self.k
        idx = (self.local_index + moves) % len(self.allowed_slots)
        return self.allowed_slots[idx]

    def __repr__(self):
        return (f"{self.name}(set={self.set_name}, weight={self.weight}, "
                f"k={self.k}, t0={self.t0}, start_pos={self.start_pos}, "
                f"allowed={self.allowed_slots}, priority={self.priority})")


class Metrics:
    """Collection and analysis of simulation metrics"""
    def __init__(self, num_slots, num_ticks):
        self.num_slots = num_slots
        self.num_ticks = num_ticks
        self.collision_count = 0  # number of collision attempts
        self.collision_prevented = 0  # number of prevented collisions
        self.slot_usage = [0] * num_slots  # how many ticks each slot was occupied
        self.slot_weight_sum = [0] * num_slots  # total weight in slot over all ticks
        self.element_wait_times = defaultdict(list)  # wait time per element
        self.history = []  # for post-analysis

    def record_slot_usage(self, slots, weights, t):
        for i in range(self.num_slots):
            if slots[i]:
                self.slot_usage[i] += 1
                self.slot_weight_sum[i] += weights[i]

    def record_collision(self, prevented=False):
        self.collision_count += 1
        if prevented:
            self.collision_prevented += 1

    def record_wait(self, element_name, t):
        self.element_wait_times[element_name].append(t)

    def get_summary(self):
        total_slots_ticks = self.num_slots * self.num_ticks
        occupied_ticks = sum(self.slot_usage)
        
        summary = {
            'total_ticks': self.num_ticks,
            'slot_utilization': occupied_ticks / total_slots_ticks * 100 if total_slots_ticks > 0 else 0,
            'avg_slot_utilization': mean(self.slot_usage) / self.num_ticks * 100,
            'collision_rate': self.collision_count / self.num_ticks if self.num_ticks > 0 else 0,
            'prevention_rate': self.collision_prevented / self.collision_count * 100 if self.collision_count > 0 else 0,
            'slot_usage_per_slot': [u / self.num_ticks * 100 for u in self.slot_usage],
            'avg_weight_per_slot': [w / max(1, self.slot_usage[i]) for i, w in enumerate(self.slot_weight_sum)],
            'element_avg_wait': {name: mean(waits) for name, waits in self.element_wait_times.items() if waits},
        }
        return summary


def print_initial(elements, num_slots, num_ticks):
    print("\n=== INITIAL PARAMETERS ===")
    print("Name | Set | Weight | k | t0 | StartPos | Priority | AllowedSlots")
    print("-" * 85)
    for e in elements:
        print(f"{e.name:4} | {e.set_name:3} | {e.weight:6} | {e.k:2} | {e.t0:2} | "
              f"{e.start_pos:8} | {e.priority:8} | {e.allowed_slots}")
    print(f"\nnum_slots = {num_slots}")
    print(f"num_ticks = {num_ticks}")


def simulate(elements, num_slots, num_ticks, allow_overlap=True, global_shift=True, real_time=True, time_sleep=0.1):
    metrics = Metrics(num_slots, num_ticks)
    history = []
    
    # Indexing elements for quick access
    elem_by_name = {e.name: e for e in elements}

    for t in range(num_ticks):
        slots = [[] for _ in range(num_slots)]  # element names in slot
        slot_elements = [[] for _ in range(num_slots)]  # element objects in slot
        weights = [0] * num_slots
        shift = t if global_shift else 0

        # Sort by priority: elements with higher priority occupy the slot first
        sorted_elements = sorted(elements, key=lambda e: (-e.priority, e.name))

        for e in sorted_elements:
            if t < e.t0:
                local_pos = e.start_pos
            else:
                local_pos = e.local_position(t)
            pos = (local_pos + shift) % num_slots

            # [!] FIXED OVERLAP CHECK
            if not allow_overlap:
                conflict = False
                for existing_elem in slot_elements[pos]:
                    # Conflict only if different sets
                    if existing_elem.set_name != e.set_name:
                        conflict = True
                        break
                
                if conflict:
                    metrics.record_collision(prevented=True)
                    e.block_count += 1
                    e.wait_time += 1
                    metrics.record_wait(e.name, t)
                    continue  # element does not occupy slot in this tick

            # Successful placement
            slots[pos].append(e.name)
            slot_elements[pos].append(e)
            weights[pos] += e.weight
            e.move_count += 1
            # Reset wait time on successful move
            if e.wait_time > 0:
                e.wait_time = 0

        metrics.record_slot_usage(slots, weights, t)
        history.append((slots, weights))

        if real_time:
            print(f"\nt={t:2d} | shift={shift:2d} | ", end="")
            for i in range(num_slots):
                if slots[i]:
                    s = "+".join(slots[i])
                    w = weights[i]
                    color = COLORS[i % len(COLORS)]
                    print(f"{color}{s}({w}){RESET} ", end="")
                else:
                    print("0 ", end="")
            print()
            time.sleep(time_sleep)
    
    metrics.history = history
    return history, metrics


def print_table(history):
    print("\n=== TABLE (time series) ===")
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
    
    print("\n" + "="*60)
    print("[STATS] ANALYTICAL REPORT")
    print("="*60)
    
    print(f"\n- General Metrics:")
    print(f"   * Simulation Duration: {summary['total_ticks']} ticks")
    print(f"   * Slot Utilization (total): {summary['slot_utilization']:.2f}%")
    print(f"   * Slot Utilization (average per slot): {summary['avg_slot_utilization']:.2f}%")
    
    print(f"\n- Conflicts:")
    print(f"   * Total collision attempts: {metrics.collision_count}")
    print(f"   * Prevented collisions: {metrics.collision_prevented} ({summary['prevention_rate']:.1f}%)")
    print(f"   * Collision rate: {summary['collision_rate']:.3f} per tick")
    
    print(f"\n- Slot Load:")
    print(f"   {'Slot':<6} {'Load, %':<12} {'Avg Weight':<10} {'Visualization'}")
    print(f"   {'-'*50}")
    for i in range(metrics.num_slots):
        bar_len = int(summary['slot_usage_per_slot'][i] / 5)
        bar = "█" * bar_len
        print(f"   {i:<6} {summary['slot_usage_per_slot'][i]:>6.2f}%      "
              f"{summary['avg_weight_per_slot'][i]:>6.2f}      {bar}")
    
    print(f"\n- Element Statistics:")
    print(f"   {'Element':<8} {'Set':<6} {'Moves':<6} {'Blocks':<10} {'Avg Wait'}")
    print(f"   {'-'*55}")
    for e in sorted(elements, key=lambda x: x.set_name + x.name):
        avg_wait = summary['element_avg_wait'].get(e.name, 0)
        print(f"   {e.name:<8} {e.set_name:<6}   {e.move_count:<6} "
              f"{e.block_count:<10} {avg_wait:>6.2f} ticks")
    
    # [SEARCH] Recommendations based on metrics
    print(f"\n- Recommendations:")
    if summary['slot_utilization'] < 30:
        print("   [WARN] Low utilization: you can decrease num_slots or add elements")
    elif summary['slot_utilization'] > 85:
        print("   [WARN] High load: risk of deadlocks, consider increasing num_slots")
    
    if summary['collision_rate'] > 0.5:
        print("   [WARN] Frequent collisions: check k and allowed_slots parameters for balancing")
    
    max_wait_elem = max(summary['element_avg_wait'].items(), key=lambda x: x[1], default=None)
    if max_wait_elem and max_wait_elem[1] > summary['total_ticks'] * 0.3:
        print(f"   [WARN] Element {max_wait_elem[0]} waits too long: increase priority or expand allowed_slots")
    
    print("="*60 + "\n")


# =========================
# SIMULATION SETUP
# =========================
if __name__ == "__main__":
    num_slots = 6
    num_ticks = 12
    time_sleep = 0.2  # delay between ticks in real-time mode

    # Allowed slots sets for different groups
    A_slots = [0, 4, 5]
    B_slots = [2, 3]
    D_slots = [1]

    elements = [
        # Set A (cyclically by [0,4,5])
        Element("A1", k=2, start_pos=0, allowed_slots=A_slots, set_name="A", priority=10),
        Element("A2", k=1, start_pos=4, allowed_slots=A_slots, set_name="A", priority=5),
        Element("A3", k=3, start_pos=5, allowed_slots=A_slots, set_name="A", priority=8),

        # Set B (cyclically by [2,3])
        Element("B1", k=1, start_pos=2, allowed_slots=B_slots, set_name="B", priority=7),
        Element("B2", k=2, start_pos=3, allowed_slots=B_slots, set_name="B", priority=3),

        # Set D (fixed slot 1)
        Element("D1", k=1, start_pos=1, allowed_slots=D_slots, set_name="D", priority=15, weight=2),
    ]

    # Simulation parameters
    allow_overlap = False   # [x] forbid overlap between different sets
    global_shift = True     # [v] enable global shift (system rotation)
    real_time_mode = True   # [v] step-by-step output with delay

    # Launch
    print_initial(elements, num_slots, num_ticks)
    
    history, metrics = simulate(
        elements, num_slots, num_ticks,
        allow_overlap=allow_overlap,
        global_shift=global_shift,
        real_time=real_time_mode,
        time_sleep=time_sleep
    )
    
    print_table(history)
    print_metrics(metrics, elements)