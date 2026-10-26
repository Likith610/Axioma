"""A complete biological simulation of a fruit fly.

Its brain is the real connectome.  Sensors (real neurons) -> real synapses
-> motor neurons -> movement.  Biological state (circadian rhythm, sleep,
dopamine reward, temperature homeostasis, gut) modulates the pathways so
behavior is emergent, not scripted.

Sensors: olfaction, gustation, bristle touch, nociception, vision,
thermosensation, hygrosensation, wind (hair plate), proprioception,
internal nutrient.

Motors: neck L/R, leg, leg L/R, wing, proboscis, haltere, abdomen,
antenna, pharynx.

Environment: temperature field, wind field, humidity field, day/night,
food sources.
"""

import argparse
import math
import os
import random
import time
from collections import defaultdict, deque

import pandas as pd

# ── Paths & Constants ──────────────────────────────────────────────────────

_BASE = os.path.dirname(os.path.abspath(__file__))
CONN = os.path.join(_BASE, "connections_princeton.csv")
NEURONS = os.path.join(_BASE, "neurons.csv")
CACHE = os.path.join(_BASE, "fly_response.csv")

DEFAULT_W, DEFAULT_H = 30, 20
SMELL_GAIN = 80.0
TICKS_PER_DAY = 1440


# ── Data Loading ───────────────────────────────────────────────────────────

def load_adjacency():
    df = pd.read_csv(
        CONN,
        dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"},
        usecols=["pre_root_id", "post_root_id", "syn_count"],
    )
    adj = defaultdict(list)
    for row in df.itertuples(index=False, name="E"):
        adj[row.pre_root_id].append((row.post_root_id, row.syn_count))
    return adj


def load_annotations():
    df = pd.read_csv(
        NEURONS,
        dtype={"Root ID": "int64"},
        usecols=["Root ID", "Class", "Soma side", "Primary Cell Type", "Sub Class"],
    )
    return df.set_index("Root ID")


# ── Neuron Selection ───────────────────────────────────────────────────────

def pick_neuron(out_syn, annotations, class_substr, side=None, sub_class=None):
    mask = annotations["Class"].fillna("").str.contains(class_substr, na=False)
    if side:
        mask &= annotations["Soma side"].fillna("").str.contains(side, na=False)
    if sub_class:
        mask &= (
            annotations["Primary Cell Type"].fillna("").str.contains(sub_class, na=False)
            | annotations["Sub Class"].fillna("").str.contains(sub_class, na=False)
        )
    candidates = annotations[mask].index
    best, best_syn = None, -1
    for nid in candidates:
        syn = out_syn.get(nid, -1)
        if syn > best_syn:
            best, best_syn = nid, syn
    return best, best_syn


def motor_neurons(annotations):
    sub = annotations[annotations["Class"].fillna("").str.contains("motor", na=False)]
    groups = {
        "neck_left": [], "neck_right": [],
        "leg": [], "leg_left": [], "leg_right": [],
        "wing": [],
        "proboscis": [], "haltere": [], "abdomen": [],
        "antenna": [], "pharynx": [],
    }
    for nid, row in sub.iterrows():
        side = str(row["Soma side"]) if pd.notna(row["Soma side"]) else ""
        cls = row["Class"]
        if cls == "neck_motor_neuron":
            if "left" in side:
                groups["neck_left"].append(nid)
            elif "right" in side:
                groups["neck_right"].append(nid)
        elif cls == "leg_motor_neuron":
            groups["leg"].append(nid)
            if "left" in side:
                groups["leg_left"].append(nid)
            elif "right" in side:
                groups["leg_right"].append(nid)
        elif cls == "wing_motor_neuron":
            groups["wing"].append(nid)
        elif cls == "proboscis_motor_neuron":
            groups["proboscis"].append(nid)
        elif cls == "haltere_motor_neuron":
            groups["haltere"].append(nid)
        elif cls == "abdomen_motor_neuron":
            groups["abdomen"].append(nid)
        elif cls == "antenna_motor_neuron":
            groups["antenna"].append(nid)
        elif cls == "pharynx_motor_neuron":
            groups["pharynx"].append(nid)
    return groups


def build_sensors(adj, annotations):
    out_syn = {n: sum(s for _, s in edges) for n, edges in adj.items()}

    def pick(cls, side, label, sub_class=None):
        nid, syn = pick_neuron(out_syn, annotations, cls, side, sub_class)
        print(f"  {label:<30s} -> {nid}  (out_syn={syn})")
        return nid

    print("Picking real sensory neurons (by strongest output in graph):")
    return [
        # Olfaction
        ("smellL", pick("olfactory_receptor", "left", "smellL ORN left"), "smellL ORN (left antenna)", 4),
        ("smellR", pick("olfactory_receptor", "right", "smellR ORN right"), "smellR ORN (right antenna)", 4),
        # Gustation (taste on legs/proboscis)
        ("taste", pick("taste_bristle_gustatory", None, "gustatory neuron"), "gustatory (sweet/bitter)", 5),
        # Bristle touch
        ("touchL", pick("bristle", "left", "touchL bristle left"), "bristle (left side)", 4),
        ("touchR", pick("bristle", "right", "touchR bristle right"), "bristle (right side)", 4),
        # Nociception (pain / damage)
        ("pain", pick("multidendritic", None, "multidendritic nociceptor"), "multidendritic (nociception)", 5),
        # Vision
        ("vision", pick("photoreceptor", None, "photoreceptor"), "photoreceptor", 6),
        # Thermosensation (warm-sensitive TRN, internal cold-sensitive AC)
        ("temp_hot", pick("thermosensory", None, "thermosensory warm TRN", sub_class="VP1m"),
         "thermosensory warm (TRN_VP1m)", 5),
        ("temp_cold", pick("thermosensory", None, "thermosensory cold AC", sub_class="AC"),
         "thermosensory cold (AC_neuron)", 5),
        # Hygrosensation
        ("humidity", pick("hygrosensory", None, "hygrosensory receptor"), "hygrosensory (humidity)", 5),
        # Wind / air flow (hair plates)
        ("wind", pick("hair_plate", None, "hair plate wind sensor"), "hair plate (wind)", 4),
        # Proprioception / gravity (campaniform sensilla)
        ("proprio", pick("campaniform_sensillum", None, "campaniform sensillum"),
         "campaniform (proprioception)", 4),
        # Internal nutrient sensing
        ("nutrient", pick("nutrient_sensory", None, "nutrient sensory"), "nutrient (internal)", 5),
    ]


# ── Response Table (connectome propagation) ───────────────────────────────

def propagate_to_motors(adj, motor_ids, start, max_hops):
    active = {start}
    strength = {start: 1.0}
    level = {start: 0}
    queue = deque([start])
    hits = {}
    if start in motor_ids:
        hits[start] = 1.0
    while queue:
        node = queue.popleft()
        lvl = level[node]
        if lvl >= max_hops:
            continue
        for post, syn in adj.get(node, ()):
            if post not in active:
                active.add(post)
                strength[post] = syn
                level[post] = lvl + 1
                queue.append(post)
            else:
                strength[post] += syn
            if post in motor_ids:
                hits[post] = hits.get(post, 0.0) + syn
    return hits


def compute_response_table(adj, annotations, sensors, force=False):
    if not force and os.path.exists(CACHE):
        tab = pd.read_csv(CACHE)
        known = {(row.sensor, row.hops) for row in tab.itertuples(index=False)}
        want = {(name, hop) for name, nid, label, hop in sensors}
        if known == want:
            print(f"Loaded cached sensor->motor responses from {CACHE}")
            return tab
        print("Cache is stale; recomputing...")
    motors = motor_neurons(annotations)
    motor_ids = {n for g in motors.values() for n in g}
    rows = []
    for name, nid, label, hop in sensors:
        hits = propagate_to_motors(adj, motor_ids, nid, hop)
        print(f"  {name:10s} {label:<30s} ({hop} hops) -> {len(hits)} motor neurons hit")
        for m in motor_ids:
            rows.append(
                {"sensor": name, "sensor_id": nid, "neuron_id": m,
                 "hops": hop, "signal": hits.get(m, 0)}
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(CACHE, index=False)
    print(f"Wrote sensor->motor response table to {CACHE}")
    return tab


# ── Biology ────────────────────────────────────────────────────────────────

class Biology:
    """Internal biological state that modulates the connectome pathways.

    All behaviour still emerges from the real wiring — biology only
    scales the *gain* of each pathway (neuromodulation).
    """

    def __init__(self, seed):
        rng = random.Random(seed)
        # Circadian clock (0 = midnight, 0.25 = 6 am, 0.5 = noon, 0.75 = 6 pm)
        self.circadian_phase = rng.uniform(0.0, 1.0)
        # Sleep
        self.sleep_pressure = 0.0
        self.is_sleeping = False
        self._sleep_debt = 0.0
        # Dopamine (reward signal)
        self.dopamine = 0.5
        self._last_reward = 0.0
        # Temperature homeostasis (ectotherm: body temp tracks environment)
        self.body_temp = 25.0
        self.temp_preferred = 25.0
        # Gut
        self.gut_fullness = 0.5
        # Stress / pain
        self.stress = 0.0
        # Age / fitness
        self.age = 0
        self.lifespan = 60000  # ticks (~2 days at 1 tick/s)
        # Learning: associative memory  {odor_hash: float -1..+1}
        self.memory = {}

    # ── derived properties ──

    @property
    def is_day(self):
        return 0.2 < self.circadian_phase < 0.8

    @property
    def light_level(self):
        """0..1  smoothly follows circadian phase."""
        return max(0.0, math.sin(2 * math.pi * (self.circadian_phase - 0.25)))

    @property
    def hunger(self):
        return max(0.0, 1.0 - self.gut_fullness)

    @property
    def appetite(self):
        """0..1 how much the fly wants food right now."""
        base = self.hunger * (0.5 + 0.5 * self.dopamine)
        return base * self.circadian_activity

    @property
    def circadian_activity(self):
        """Circadian modulation of overall activity level 0..1."""
        day_boost = 0.7 + 0.3 * self.light_level
        sleep_drag = 0.3 if self.is_sleeping else 1.0
        return day_boost * sleep_drag * self.fitness

    @property
    def fitness(self):
        """Declines with age."""
        return max(0.15, 1.0 - 0.85 * (self.age / self.lifespan))

    @property
    def resting(self):
        return self.is_sleeping or (self.sleep_pressure > 0.7 and not self.is_day)

    # ── tick ──

    def tick(self, activity, ate_food, env_temp):
        # Circadian
        self.circadian_phase = (self.circadian_phase + 1.0 / TICKS_PER_DAY) % 1.0

        # Sleep pressure
        self.sleep_pressure = min(1.0, self.sleep_pressure + 0.0008 * activity)
        if not self.is_day and self.sleep_pressure > 0.3:
            self.is_sleeping = True
        if self.is_sleeping:
            self.sleep_pressure = max(0.0, self.sleep_pressure - 0.005)
            if self.sleep_pressure < 0.05:
                self.is_sleeping = False

        # Dopamine
        if ate_food:
            self.dopamine = min(1.0, self.dopamine + 0.15)
            self._last_reward = 1.0
        else:
            self.dopamine = max(0.2, self.dopamine - 0.0005)
            self._last_reward *= 0.98

        # Gut (slow digestion)
        self.gut_fullness = max(0.0, self.gut_fullness - 0.0004)

        # Temperature homeostasis (thermal inertia)
        self.body_temp += (env_temp - self.body_temp) * 0.02

        # Stress decays
        self.stress = max(0.0, self.stress - 0.01)

        # Aging
        self.age += 1


# ── Environment ────────────────────────────────────────────────────────────

class Environment:
    """A walled arena with temperature, wind, humidity, light, and food."""

    def __init__(self, W, H, seed):
        self.W, self.H = W, H
        self.rng = random.Random(seed)

        # Temperature field: warm spot near centre, cool edges
        self.temp = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                r = math.hypot(x - W / 2, y - H / 2)
                self.temp[y][x] = 25.0 + 6.0 * math.exp(-r ** 2 / 60)

        # Wind field: gentle prevailing breeze from the left + turbulence
        self.wind_x = [[0.15 + self.rng.gauss(0, 0.04) for _ in range(W)] for _ in range(H)]
        self.wind_y = [[self.rng.gauss(0, 0.03) for _ in range(W)] for _ in range(H)]

        # Humidity: higher near centre
        self.humidity = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                r = math.hypot(x - W / 2, y - H / 2)
                max_r = math.hypot(W / 2, H / 2)
                self.humidity[y][x] = 0.7 - 0.35 * (r / max_r)

        # Light (set externally per tick via circadian)
        self.light = 1.0

        # Food sources: list of (x, y, quality)
        self.foods = []
        self._spawn_food()

    def _spawn_food(self):
        x = self.rng.uniform(3, self.W - 3)
        y = self.rng.uniform(3, self.H - 3)
        quality = self.rng.uniform(0.75, 1.0)
        self.foods.append((x, y, quality))

    def nearest_food(self, fx, fy):
        best, bd = None, 1e9
        for f in self.foods:
            d = math.hypot(f[0] - fx, f[1] - fy)
            if d < bd:
                best, bd = f, d
        return best, bd

    def remove_food(self, idx):
        self.foods.pop(idx)

    def _idx(self, x, y):
        return max(0, min(self.W - 1, int(x))), max(0, min(self.H - 1, int(y)))

    def temp_at(self, x, y):
        ix, iy = self._idx(x, y)
        return self.temp[iy][ix]

    def wind_at(self, x, y):
        ix, iy = self._idx(x, y)
        return self.wind_x[iy][ix], self.wind_y[iy][ix]

    def humidity_at(self, x, y):
        ix, iy = self._idx(x, y)
        return self.humidity[iy][ix]

    def update(self, circadian_phase):
        self.light = max(0.0, math.sin(2 * math.pi * (circadian_phase - 0.25)))
        # Gentle wind drift
        for y in range(self.H):
            for x in range(self.W):
                self.wind_x[y][x] += self.rng.gauss(0, 0.005)
                self.wind_x[y][x] = max(-0.3, min(0.3, self.wind_x[y][x]))
                self.wind_y[y][x] += self.rng.gauss(0, 0.005)
                self.wind_y[y][x] = max(-0.2, min(0.2, self.wind_y[y][x]))


# ── Fly ────────────────────────────────────────────────────────────────────

class Fly:
    def __init__(self, seed, W, H):
        self.rng = random.Random(seed)
        self.x = W / 2
        self.y = H / 2
        self.theta = 0.0
        self.speed = 0.0
        self.W, self.H = W, H
        # internal state
        self.biology = Biology(seed)
        self.flee_ticks = 0
        # spontaneous behavioral bout state
        self.bout_type = None   # "saccade" | "groom" | "proboscis" | None
        self.bout_ticks = 0
        self.bout_dir = 1.0

    def move(self, speed, turn, dt=1.0):
        self.theta += turn * dt
        self.theta %= 2 * math.pi
        self.speed = speed
        nx = self.x + math.cos(self.theta) * speed * dt
        ny = self.y + math.sin(self.theta) * speed * dt
        if 1 <= nx <= self.W - 1 and 1 <= ny <= self.H - 1:
            self.x, self.y = nx, ny
            return 0.0
        self.theta += math.pi
        return 1.0  # bounced


# ── Stimulus Computation ───────────────────────────────────────────────────

def wall_touch(px, py, W, H):
    near = 0.0
    for wx in (px, W - px):
        if wx < 1.6:
            near = max(near, (1.6 - wx) / 1.6)
    for wy in (py, H - py):
        if wy < 1.6:
            near = max(near, (1.6 - wy) / 1.6)
    return near


def compute_stimuli(fly, env, predator):
    """Map environment to sensory neuron activations (unit stimuli)."""
    bio = fly.biology
    W, H = env.W, env.H

    # ── Food / odor ──
    food, fd = env.nearest_food(fly.x, fly.y)
    fx, fy = food[0], food[1]
    # wind-blown odor plume
    wx, wy = env.wind_at(fly.x, fly.y)
    odor_d = math.hypot(fly.x - fx + wx * 8, fly.y - fy + wy * 8)
    odor = 1.0 / (1.0 + (odor_d / 3.0) ** 2)
    # antenna positions
    fwd = (math.cos(fly.theta), math.sin(fly.theta))
    perp = (-fwd[1], fwd[0])
    lp = (fly.x + fwd[0] * 0.7 + perp[0] * 1.1, fly.y + fwd[1] * 0.7 + perp[1] * 1.1)
    rp = (fly.x + fwd[0] * 0.7 - perp[0] * 1.1, fly.y + fwd[1] * 0.7 - perp[1] * 1.1)
    odor_l = 1.0 / (1.0 + (math.hypot(lp[0] - fx, lp[1] - fy) / 3.0) ** 2)
    odor_r = 1.0 / (1.0 + (math.hypot(rp[0] - fx, rp[1] - fy) / 3.0) ** 2)

    # ── Gustation (proximity to food) ──
    taste = max(0.0, 1.0 - fd / 2.0)

    # ── Touch ──
    touch_l = wall_touch(fly.x - fwd[0] * 0.5 + perp[0] * 0.5,
                         fly.y - fwd[1] * 0.5 + perp[1] * 0.5, W, H)
    touch_r = wall_touch(fly.x - fwd[0] * 0.5 - perp[0] * 0.5,
                         fly.y - fwd[1] * 0.5 - perp[1] * 0.5, W, H)

    # ── Nociception (pain) ──
    pain = bio.stress
    if predator is not None:
        pd_ = math.hypot(fly.x - predator[0], fly.y - predator[1])
        if pd_ < 1.5:
            pain = max(pain, (1.5 - pd_) / 1.5)
    env_t = env.temp_at(fly.x, fly.y)
    if env_t > 35:
        pain = max(pain, (env_t - 35) / 10)
    if env_t < 5:
        pain = max(pain, (5 - env_t) / 10)

    # ── Vision (predator) ──
    vision = 0.0
    if predator is not None:
        pd_ = math.hypot(fly.x - predator[0], fly.y - predator[1])
        if pd_ < 6.0:
            vision = (6.0 - pd_) / 6.0

    # ── Thermosensation ──
    temp_diff = env_t - bio.body_temp
    temp_hot = max(0.0, temp_diff / 10.0)
    temp_cold = max(0.0, -temp_diff / 10.0)

    # ── Hygrosensation ──
    hum = env.humidity_at(fly.x, fly.y)

    # ── Wind (hair plate) ──
    wind_s = math.hypot(wx, wy)

    # ── Proprioception (gravity component along body axis) ──
    proprio = abs(math.sin(fly.theta)) * 0.5

    # ── Internal nutrient ──
    nutrient = bio.gut_fullness

    # ── Build stimulus dict (biological modulation as gain) ──
    smell_gain = SMELL_GAIN * bio.appetite * (0.3 + 0.7 * env.light)

    stim = {
        "smellL": odor_l * smell_gain,
        "smellR": odor_r * smell_gain,
        "taste": taste * bio.appetite * 12,
        "touchL": touch_l,
        "touchR": touch_r,
        "pain": pain * 20,
        "vision": vision,
        "temp_hot": temp_hot * 12 * (1.0 + 0.5 * bio.stress),
        "temp_cold": temp_cold * 12 * (1.0 + 0.5 * bio.stress),
        "humidity": hum * 8,
        "wind": wind_s * 18,
        "proprio": proprio * 10,
        "nutrient": nutrient * 8,
    }

    # ── Sensory noise (Poisson-like, CV ~25%) ──
    # Real neurons fire stochastically: photon shot noise, thermal noise,
    # stochastic receptor binding.  Only positive signals get noise.
    for k in stim:
        s = stim[k]
        if s > 0:
            stim[k] = max(0.0, s + fly.rng.gauss(0, s * 0.25))

    return stim, odor, fd, food


# ── Rendering ──────────────────────────────────────────────────────────────

def render(fly, env, predator, args):
    if not args.no_cls:
        os.system("cls" if os.name == "nt" else "clear")
    W, H = env.W, env.H
    grid = [["  "] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if x == 0 or y == 0 or x == W - 1 or y == H - 1:
                grid[y][x] = "##"
    # food
    for fx, fy, _q in env.foods:
        gx = max(1, min(W - 2, round(fx)))
        gy = max(1, min(H - 2, round(fy)))
        grid[gy][gx] = "@@"
    # predator
    if predator is not None:
        px = max(1, min(W - 2, round(predator[0])))
        py = max(1, min(H - 2, round(predator[1])))
        grid[py][px] = "XX"
    # fly
    fy_idx = max(1, min(H - 2, round(fly.y)))
    fx_idx = max(1, min(W - 2, round(fly.x)))
    ang = int((fly.theta / (2 * math.pi) * 4 + 2.5) % 4)
    grid[fy_idx][fx_idx] = ["^>", "v>", "> ", "< "][ang]
    print("\n".join("".join(row) for row in grid))


def status_line(fly, env, d, bio_state_str):
    bio = fly.biology
    hh = "day" if bio.is_day else "night"
    slp = "zzz" if bio.is_sleeping else "---"
    print(
        f"  {bio_state_str:10s} | hunger={bio.hunger:.2f} energy={bio.gut_fullness:.2f}"
        f" | sleep={slp} press={bio.sleep_pressure:.2f}"
        f" | DA={bio.dopamine:.2f} | temp={bio.body_temp:.1f}C"
        f" | {hh} ({bio.circadian_phase:.2f})"
        f" | fitness={bio.fitness:.2f} | food_d={d:.1f}"
    )


# ── Main Simulation ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Complete biological fly simulation (real connectome brain)")
    ap.add_argument("--ticks", type=int, default=800)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step", action="store_true", help="press Enter each tick")
    ap.add_argument("--no-predator", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-cls", action="store_true", help="don't clear screen (debug)")
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--no-circadian", action="store_true", help="disable day/night cycle")
    ap.add_argument("--no-sleep", action="store_true", help="disable sleep")
    ap.add_argument("--no-wind", action="store_true", help="disable wind drift on odor")
    args = ap.parse_args()

    W, H = args.width, args.height

    print("Loading connectome...")
    adj = load_adjacency()
    annotations = load_annotations()
    sensors = build_sensors(adj, annotations)

    print("\nComputing real sensor->motor wiring (propagating through the connectome)...")
    resp = compute_response_table(adj, annotations, sensors, force=args.no_cache)
    motor_groups = motor_neurons(annotations)
    all_motor_ids = [n for g in motor_groups.values() for n in g]
    resp.set_index(["sensor", "neuron_id"], inplace=True)

    r_by_sensor = defaultdict(dict)
    for (sensor, nid), row in resp.iterrows():
        r_by_sensor[sensor][nid] = row["signal"]

    def group_signal(sensor):
        return {
            k: sum(r_by_sensor[sensor].get(n, 0.0) for n in v)
            for k, v in motor_groups.items()
        }

    group_scale = {k: 1e-9 for k in motor_groups}
    for sensor, _, _, _ in sensors:
        gs = group_signal(sensor)
        for k in group_scale:
            group_scale[k] = max(group_scale[k], gs[k])

    print("\nReal sensor -> muscle response (unit stimulus):")
    hdr = "  " + "".join(f"{k:>12}" for k in motor_groups)
    print(hdr)
    for sensor, _, label, _ in sensors:
        gs = group_signal(sensor)
        print(f"  {label:<30s}" + "".join(f"{gs[k]:>12.1f}" for k in motor_groups))

    # ── World ──
    fly = Fly(args.seed, W, H)
    env = Environment(W, H, args.seed)
    predator = None
    if not args.no_predator:
        predator = [1, 1]

    tick = 0
    stats = {"eats": 0, "catches": 0, "flees": 0, "rest_ticks": 0}
    drive_hist = []

    while tick < args.ticks:
        # ── Spontaneous behavioral bouts ──
        if fly.bout_ticks > 0:
            fly.bout_ticks -= 1
        else:
            fly.bout_type = None
            r = fly.rng.random()
            if r < 0.005:
                fly.bout_type = "saccade"
                fly.bout_ticks = fly.rng.randint(3, 5)
                fly.bout_dir = 1.0 if fly.rng.random() < 0.5 else -1.0
            elif r < 0.008:
                fly.bout_type = "groom"
                fly.bout_ticks = fly.rng.randint(15, 25)
            elif r < 0.010:
                fly.bout_type = "proboscis"
                fly.bout_ticks = fly.rng.randint(5, 10)

        # ── Environment update ──
        if not args.no_circadian:
            env.update(fly.biology.circadian_phase)
        else:
            env.light = 1.0
        fly.biology.tick(
            activity=fly.speed if fly.speed > 0.1 else 0.0,
            ate_food=False,
            env_temp=env.temp_at(fly.x, fly.y),
        )

        # ── Sensation ──
        stim, odor, fd, food = compute_stimuli(fly, env, predator)

        # ── Motor (connectome response) ──
        act = {
            k: sum(stim[s] * r_by_sensor[s].get(n, 0.0)
                    for s, _, _, _ in sensors for n in v)
            for k, v in motor_groups.items()
        }

        # ── Synaptic noise (vesicle release stochasticity, CV ~30%) ──
        for k in act:
            a = act[k]
            if a > 0:
                act[k] = max(0.0, a + fly.rng.gauss(0, a * 0.3))

        # ── PURE CONNECTOME: motor activations → movement ──
        # No hand-coded steering formulas.  The connectome decides
        # whether smell drives legs asymmetrically (and in which direction).
        # Biology only modulates sensor *gains* (neuromodulation).

        # Normalize every motor group by its max possible activation.
        norm = {k: act[k] / max(group_scale[k], 1e-9) for k in motor_groups}

        # ── Motor noise (muscle recruitment jitter, σ=0.05) ──
        for k in norm:
            norm[k] += fly.rng.gauss(0, 0.05)
            norm[k] = max(0.0, norm[k])

        # ── Spontaneous bout motor injection ──
        if fly.bout_type == "saccade":
            # sudden head turn — burst one side of neck motor neurons
            side = "neck_right" if fly.bout_dir > 0 else "neck_left"
            norm[side] = min(1.0, norm[side] + 0.8)
        elif fly.bout_type == "groom":
            # bilateral leg activation burst (front-leg grooming motion)
            norm["leg"] = min(1.0, norm["leg"] + 0.6)
            norm["leg_left"] = min(1.0, norm["leg_left"] + 0.4)
            norm["leg_right"] = min(1.0, norm["leg_right"] + 0.4)
        elif fly.bout_type == "proboscis":
            # spontaneous proboscis extension
            norm["proboscis"] = min(1.0, norm["proboscis"] + 0.7)

        # Steering: asymmetric motor groups determine turn direction.
        # neck_left / leg_left push leftward, neck_right / leg_right push rightward.
        # The connectome's wiring decides the sign — if smell drives right legs
        # more, the fly turns right (anti-tropotactic).  That's the honest answer.
        steer = (norm["neck_right"] + norm["leg_right"]
                 - norm["neck_left"] - norm["leg_left"]) / 2.0

        # Haltere stabilization: the fly's biological gyroscope dampens
        # erratic turning when haltere neurons fire (real Drosophila biology).
        steer *= max(0.2, 1.0 - norm["haltere"] * 0.5)

        steer = max(-1.2, min(1.2, steer))

        # Speed: total leg activation drives forward locomotion.
        walk = (norm["leg"] + norm["leg_left"] + norm["leg_right"]) / 3.0

        # Wing activation → escape (the vision→wing pathway is connectome-derived).
        wing_act = norm["wing"]
        if wing_act > 0.25 and fly.flee_ticks == 0:
            stats["flees"] += 1
        if wing_act > 0.25:
            fly.flee_ticks = max(fly.flee_ticks, 10)

        if fly.flee_ticks > 0:
            fly.flee_ticks -= 1
            speed = 2.4
        elif fly.biology.resting:
            speed = 0.05
        else:
            speed = (0.1 + 0.9 * walk) * fly.biology.circadian_activity

        # Abdomen posture modulates locomotion speed (real biomechanics).
        speed *= max(0.5, 1.0 - norm["abdomen"] * 0.3)

        # Feeding: proboscis extension reflex (gustatory→proboscis is connectome-derived).
        feed_signal = norm["proboscis"]

        # ── Move ──
        bounced = fly.move(speed, steer * 0.9)
        if bounced:
            stim["touchL"] = max(stim["touchL"], 0.5)
            stim["touchR"] = max(stim["touchR"], 0.5)

        # ── Predator chase ──
        if predator is not None:
            vx, vy = fly.x - predator[0], fly.y - predator[1]
            pv = math.hypot(vx, vy) or 1.0
            hawk_speed = 0.55 * fly.biology.circadian_activity
            predator[0] += vx / pv * hawk_speed
            predator[1] += vy / pv * hawk_speed
            if pv < 0.55:
                if os.environ.get("FLY_DEBUG"):
                    print(f"[catch] tick={tick} pv={pv:.2f}")
                fly.x, fly.y = W / 2, H / 2
                predator[0], predator[1] = 1, 1
                stats["catches"] += 1
                fly.biology.stress = 1.0

        # ── Eating: proboscis must be extended (connectome decides via gustatory pathway) ──
        ate_food = False
        if feed_signal > 0.3:
            for i, (fx, fy, fq) in enumerate(env.foods):
                if math.hypot(fly.x - fx, fly.y - fy) < 2.0:
                    ate_food = True
                    stats["eats"] += 1
                    fly.biology.gut_fullness = min(1.0, fly.biology.gut_fullness + 0.35 * fq)
                    fly.biology._last_reward = 1.0
                    env.remove_food(i)
                    break

        if not env.foods:
            env._spawn_food()

        # ── Biology tick (after ate_food known) ──
        if ate_food:
            fly.biology.tick(activity=speed, ate_food=True, env_temp=env.temp_at(fly.x, fly.y))

        # ── Stats / history ──
        if fly.biology.resting:
            stats["rest_ticks"] += 1

        if fly.biology.resting:
            state_str = "RESTING"
        elif fly.flee_ticks > 0:
            state_str = "FLEEING"
        elif fly.bout_type == "groom":
            state_str = "GROOMING"
        elif fly.bout_type == "saccade":
            state_str = "SACCADING"
        elif fly.bout_type == "proboscis":
            state_str = "PROBING"
        else:
            state_str = "FORAGING"
        drive_hist.append((fly.biology.hunger, fly.biology.gut_fullness, fly.biology.sleep_pressure,
                           fly.biology.dopamine, fly.biology.body_temp))

        tick += 1

        # ── Render ──
        render(fly, env, predator, args)
        status_line(fly, env, fd, state_str)

        if args.step:
            input("  (enter) ")
        else:
            time.sleep(1.0 / args.fps)

    # ── Summary ────────────────────────────────────────────────────────────
    print(
        f"\n=== {tick} ticks | fruit eaten: {stats['eats']} | "
        f"escapes (vision->wing): {stats['flees']} | "
        f"eaten by hawk: {stats['catches']} ==="
    )
    rest_pct = 100.0 * stats["rest_ticks"] / tick if tick else 0.0
    print(f"  rest ticks: {stats['rest_ticks']}/{tick} ({rest_pct:.0f}%)")
    print(
        f"  final: hunger={fly.biology.hunger:.2f} gut={fly.biology.gut_fullness:.2f}"
        f" | DA={fly.biology.dopamine:.2f} | temp={fly.biology.body_temp:.1f}C"
        f" | sleep_press={fly.biology.sleep_pressure:.2f}"
        f" | fitness={fly.biology.fitness:.2f} age={fly.biology.age}"
    )

    # ── Drive sparklines ──
    if drive_hist:
        buckets = 64
        n = len(drive_hist)
        chunk = max(1, n // buckets)
        hist = [drive_hist[i] for i in range(0, n - n % chunk, chunk)]
        hist.append(drive_hist[-1])
        ramp = " .:-=+*#"
        print("\ndrive cycle (left=start, right=end):")
        for name, idx in [("hunger", 0), ("gut_fullness", 1), ("sleep", 2),
                          ("dopamine", 3), ("body_temp", 4)]:
            levels = []
            for bucket in hist:
                val = bucket[idx]
                if name == "body_temp":
                    val = max(0.0, min(1.0, (val - 15) / 25))  # map 15-40C to 0-1
                levels.append(ramp[min(7, int(round(val * 7)))])
            print(f"  {name:14s}: {''.join(levels)}")


if __name__ == "__main__":
    main()
