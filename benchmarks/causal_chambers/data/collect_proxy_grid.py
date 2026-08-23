"""
Collect the proximal benchmarks with proxy informativeness (high/mid/low) as an
additional setting. 4 confounding settings x 3 levels = 12 csvs, N rows each.

The knob adds gaussian noise to the software edges that set the proxies,
Z (led_2_uv) and W (pol_1), scaled by each variable's sd in the A1=500 data.
This matches the standard proximal DGP where proxies are noisy functions of U.
X (red) is left untouched because noise there breaks the 2SLS first stage by making it a pseudo-RCT.
"""

import os
import numpy as np
import pandas as pd
import causalchamber.lab as lab

N = 1000
A1 = 500
A0 = 0
CHAMBER_ID = "lt-demo-ch4lu"
CREDENTIALS_FILE = os.path.expanduser("~/.causalchamber/cc_key.txt")
SEED = 981264976135  # confounder stream, as in all previous runs
NOISE_SEED = SEED + 7  # proxy-noise stream
OUTDIR = "proxy_grid"

# (kZ, kW) noise multipliers per level. strong_linear/low is asymmetric:
# symmetric noise there collapsed the confounding (Z drives ir_2, which
# assigns A), while W carries most of U, so heavy W noise reaches the same
# R2(U|Z,W) target with the confounding intact (in my test: 31% of effect vs 5%).
LEVEL_K = {"high": (0.0, 0.0), "mid": (0.75, 0.75), "low": (2.0, 2.0)}
CELL_K = {("strong_confounding_linear", "low"): (0.6, 2.5)}

# per-setting proxy sds from the A1=500 n=1000 data
PROXY_SD = {
    "weak_confounding_linear": dict(led_2_uv=325.0, pol_1=62.2),
    "weak_confounding_non_linear": dict(led_2_uv=1789.0, pol_1=64.5),
    "strong_confounding_linear": dict(led_2_uv=383.0, pol_1=66.2),
    "strong_confounding_non_linear": dict(led_2_uv=412.0, pol_1=62.6),
}


def clamped(x, a, b, lo, hi):
    return np.maximum(lo, np.minimum(hi, a * x + b))


def logistic(x, l, k, x0, b):
    return l / (1 + np.exp(-k * (x - x0))) + b


f1_linear = lambda x: clamped(x, -0.085, -1445, 0, 255)
f2_linear = lambda x: clamped(x, 0.0624, 0, 0, 4095)
f1_non_linear = lambda x: logistic(x, l=255, k=2e-3, x0=-19000.0, b=0)
f2_non_linear = lambda x: logistic(x, l=4095, k=1e-3, x0=35000.0, b=0)
f3_binary_cut = lambda x: (x < 5250).astype(int)
f4 = lambda x: logistic(x, l=180, k=2e-3, x0=4000, b=-90)

# Strong non-linear uses a smaller sigmoid amplitude. With l=4095 the led_2_uv
# swing dominates ir_2, so U -> A runs through Z and proxy noise destroys the
# confounding (pilot: 38/1/15% across levels). At l=1000 the confounder reaches
# ir_2 directly (corr(U, ir_2) = 0.93) and the knob leaves it intact (43/16/17%).
# Threshold 3930 is the 45th percentile of the new ir_2 distribution.
f2_non_linear_strong = lambda x: logistic(x, l=1000, k=1e-3, x0=35000.0, b=0)
f3_strong_nl = lambda x: (x < 3930).astype(int)

SETTINGS = {
    "weak_confounding_linear": (f1_linear, f2_linear, f3_binary_cut, f4, False),
    "weak_confounding_non_linear": (
        f1_non_linear,
        f2_non_linear,
        f3_binary_cut,
        f4,
        False,
    ),
    "strong_confounding_linear": (f1_linear, f2_linear, f3_binary_cut, f4, True),
    "strong_confounding_non_linear": (
        f1_non_linear,
        f2_non_linear_strong,
        f3_strong_nl,
        f4,
        True,
    ),
}


def sample_one(chamber, green, f1, f2, f3, f4, max_confounding, noise_sd, eps):
    # reset all inputs
    batch = chamber.new_batch()
    for v in ("green", "blue", "red", "led_2_uv", "pol_1", "pol_2", "led_3_uv"):
        batch.set(v, 0)
    batch.set("diode_ir_2", 2 if max_confounding else 0)
    batch.set("diode_ir_3", 2 if max_confounding else 0)
    batch.set("t_ir_2", 3 if max_confounding else 0)
    batch.set("t_ir_3", 3 if max_confounding else 0)

    # confounder LED
    batch = chamber.new_batch()
    batch.set("blue" if max_confounding else "green", green)
    batch.measure(n=1)
    df = batch.submit().iloc[0]

    # X = red, no noise
    current_ls_raw = df.current_ls_raw
    red = int(np.clip(round(f1(current_ls_raw)), 0, 255))
    batch = chamber.new_batch()
    batch.set("red", red)
    batch.measure(n=1)
    df = batch.submit().iloc[0]

    # proxies Z and W with software-edge noise
    ir_1, vis_1 = df.ir_1, df.vis_1
    led_2_uv = f2(ir_1)
    pol_1 = f4(vis_1)
    if noise_sd is not None:
        led_2_uv = led_2_uv + eps[0] * noise_sd["led_2_uv"]
        pol_1 = pol_1 + eps[1] * noise_sd["pol_1"]
    led_2_uv = int(np.clip(round(led_2_uv), 0, 4095))
    pol_1 = float(np.clip(pol_1, -180, 180))
    batch = chamber.new_batch()
    batch.set("led_2_uv", led_2_uv)
    batch.set("pol_1", pol_1)
    if max_confounding:
        pol_2 = 0.0
    else:
        pol_2 = pol_1 + 90  # crossed polariser tracks the noised pol_1;
        if pol_2 > 180:  # +90 and -90 are the same orientation
            pol_2 -= 180
    batch.set("pol_2", pol_2)
    batch.measure(n=1)
    df = batch.submit().iloc[0]

    # binary treatment from ir_2, then Y and both counterfactuals
    ir_2 = df.ir_2
    led_3_uv = int(round(f3(np.asarray(ir_2)))) * A1
    batch = chamber.new_batch()
    batch.set("led_3_uv", led_3_uv)
    batch.measure(n=1)
    ir_3 = batch.submit().iloc[0].ir_3

    batch = chamber.new_batch()
    batch.set("led_3_uv", A0)
    batch.measure(n=1)
    Y0 = batch.submit().iloc[0].ir_3

    batch = chamber.new_batch()
    batch.set("led_3_uv", A1)
    batch.measure(n=1)
    Y1 = batch.submit().iloc[0].ir_3

    return pd.DataFrame(
        {
            "green": [green],
            "current_ls_raw": [current_ls_raw],
            "red": [red],
            "vis_1": [vis_1],
            "pol_1": [pol_1],
            "pol_2": [pol_2],
            "ir_1": [ir_1],
            "led_2_uv": [led_2_uv],
            "ir_2": [ir_2],
            "led_3_uv": [led_3_uv],
            "ir_3": [ir_3],
            "Y1": [Y1],
            "Y0": [Y0],
        }
    )


os.makedirs(OUTDIR, exist_ok=True)
for setting, (f1, f2, f3, f4_, max_conf) in SETTINGS.items():
    for level in ("high", "mid", "low"):
        path = f"{OUTDIR}/out_binaryA_counterfactuals_{setting}_proxy_{level}.csv"
        done = len(pd.read_csv(path)) if os.path.exists(path) else 0
        if done >= N:
            print(f"{setting}/{level}: complete, skipping")
            continue

        kz, kw = CELL_K.get((setting, level), LEVEL_K[level])
        noise_sd = None
        if kz > 0 or kw > 0:
            noise_sd = {
                "led_2_uv": kz * PROXY_SD[setting]["led_2_uv"],
                "pol_1": kw * PROXY_SD[setting]["pol_1"],
            }

        greens = np.random.default_rng(SEED).integers(0, 256, size=N)
        noise_rng = np.random.default_rng(NOISE_SEED)
        for _ in range(done):  # keep the stream aligned on resume
            noise_rng.normal(0, 1, size=2)

        chamber = lab.Chamber(CHAMBER_ID, "standard", credentials_file=CREDENTIALS_FILE)
        for i in range(done, N):
            eps = noise_rng.normal(0, 1, size=2)
            row = sample_one(
                chamber, greens[i], f1, f2, f3, f4_, max_conf, noise_sd, eps
            )
            row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"{setting}/{level}: {i + 1}/{N}", end="\r")
        print()
