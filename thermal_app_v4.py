import streamlit as st
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as mpatches
from io import BytesIO
import pandas as pd

# ============================================================================
# MATERIAL & DIMENSION DATABASES
# ============================================================================

# Thermal conductivities (W/m·K)
CONDUIT_MATERIALS = {
    # HDPE max operating temp 82°C per Plastics Pipe Institute TN-11
    # (non-pressure applications, e.g. conduit/duct)
    "HDPE": {"k": 0.48, "max_temp": 82.0, "standard": "ASTM F2160 / PPI TN-11"},
    # PVC rated to 60°C per ASTM D2665 / NEMA TC-2
    "PVC":  {"k": 0.19, "max_temp": 60.0, "standard": "ASTM D2665 / NEMA TC-2"},
    "Steel":{"k": 50.0, "max_temp": 300.0,"standard": "ASTM A53"},
    "Fiberglass":{"k":0.35,"max_temp":120.0,"standard":"ASTM D2996"}
}

# HDPE conduit dimensions (ASTM F2160) - (OD_in, ID_in)
# IPS sizing: nominal pipe size maps to specific OD
HDPE_DIMS = {
    "3 IPS": {
        "SCH 40": (3.500, 3.068),
        "SCH 80": (3.500, 2.900),
        "SDR 11": (3.500, 3.182),
        "SDR 13.5": (3.500, 3.240),
    },
    "4 IPS": {
        "SCH 40": (4.500, 4.026),
        "SCH 80": (4.500, 3.826),
        "SDR 11": (4.500, 4.091),
        "SDR 13.5": (4.500, 4.167),
    },
    "5 IPS": {
        "SCH 40": (5.563, 5.047),
        "SCH 80": (5.563, 4.813),
        "SDR 11": (5.563, 5.057),
        "SDR 13.5": (5.563, 5.152),
    },
    "6 IPS": {
        "SCH 40": (6.625, 6.065),
        "SCH 80": (6.625, 5.761),
        "SDR 11": (6.625, 6.023),
        "SDR 13.5": (6.625, 6.139),
    },
    "8 IPS": {
        "SCH 40": (8.625, 7.981),
        "SCH 80": (8.625, 7.625),
        "SDR 11": (8.625, 7.841),
        "SDR 13.5": (8.625, 7.991),
    },
}

# PVC conduit dimensions (ASTM D2665 / NEMA TC-2) - (OD_in, ID_in)
PVC_DIMS = {
    "3 IPS": {
        "SCH 40": (3.500, 3.042),
        "SCH 80": (3.500, 2.864),
    },
    "4 IPS": {
        "SCH 40": (4.500, 4.000),
        "SCH 80": (4.500, 3.786),
    },
    "5 IPS": {
        "SCH 40": (5.563, 5.016),
        "SCH 80": (5.563, 4.768),
    },
    "6 IPS": {
        "SCH 40": (6.625, 6.031),
        "SCH 80": (6.625, 5.709),
    },
    "8 IPS": {
        "SCH 40": (8.625, 7.943),
        "SCH 80": (8.625, 7.565),
    },
}

CABLE_CONDUCTOR_MATERIALS = {
    "Copper":   {"k": 385.0},
    "Aluminum": {"k": 205.0},
}

CABLE_JACKET_MATERIALS = {
    "XLPE": {"k": 0.286},
    "EPR":  {"k": 0.250},
    "PVC":  {"k": 0.190},
    "PE":   {"k": 0.380},
}

SURROUNDING_MATERIALS = {
    "Dry Soil":    {"k": 0.25},
    "Moist Soil":  {"k": 1.00},
    "Wet Soil":    {"k": 1.50},
    "Sand":        {"k": 0.58},
    "Clay":        {"k": 1.28},
    "Concrete":    {"k": 1.40},
}

CABLE_SIZES = {
    "350 kcmil": {"copper_OD": 0.681, "jacket_OD": 0.980},
    "500 kcmil": {"copper_OD": 0.813, "jacket_OD": 1.063},
    "750 kcmil": {"copper_OD": 0.968, "jacket_OD": 1.203},
    "1000 kcmil":{"copper_OD": 1.098, "jacket_OD": 1.348},
    "1250 kcmil":{"copper_OD": 1.213, "jacket_OD": 1.470},
}

# Preset scenarios
PRESETS = {
    "Single Cable Baseline": {
        "description": "One 4\" HDPE conduit, 750 kcmil copper, moist soil",
        "grid_x": 1, "grid_y": 1,
        "h_spacing": 6.0, "v_spacing": 6.0,
        "surrounding": "Moist Soil",
        "global_T_conductor": 70.0,
        "T_ambient": 10.0,
    },
    "Typical 2×2 Install": {
        "description": "Four 4\" HDPE conduits in 2×2 grid, 750 kcmil copper, moist soil",
        "grid_x": 2, "grid_y": 2,
        "h_spacing": 6.0, "v_spacing": 6.0,
        "surrounding": "Moist Soil",
        "global_T_conductor": 70.0,
        "T_ambient": 10.0,
    },
    "Concrete Encased Bank": {
        "description": "3×2 bank in concrete encasement, 750 kcmil copper",
        "grid_x": 3, "grid_y": 2,
        "h_spacing": 7.5, "v_spacing": 7.5,
        "surrounding": "Concrete",
        "global_T_conductor": 70.0,
        "T_ambient": 15.0,
    },
}

# ============================================================================
# HELPER: GET CONDUIT DIMS
# ============================================================================

def get_conduit_dims(material, ips_size, wall_type):
    """Return (OD_m, ID_m) for given conduit spec."""
    if material == "HDPE":
        od_in, id_in = HDPE_DIMS[ips_size][wall_type]
    else:  # PVC
        od_in, id_in = PVC_DIMS[ips_size][wall_type]
    return od_in * 0.0254, id_in * 0.0254

def get_wall_types(material, ips_size):
    if material == "HDPE":
        return list(HDPE_DIMS[ips_size].keys())
    else:
        return list(PVC_DIMS[ips_size].keys())

def validate_cable_fit(conduit_radius_inner, jacket_radius, n_cables):
    """
    Check whether n_cables of jacket_radius fit inside conduit_radius_inner.
    Returns (ok: bool, message: str).
    """
    if jacket_radius >= conduit_radius_inner:
        return False, "Cable jacket diameter exceeds conduit inner diameter."
    if n_cables == 0:
        return True, ""
    # Practical fill ratio check (NEC 40% fill guideline)
    cable_area   = n_cables * np.pi * jacket_radius**2
    conduit_area = np.pi * conduit_radius_inner**2
    fill = cable_area / conduit_area
    if fill > 0.85:
        return False, (f"Cable fill ratio {fill*100:.0f}% exceeds practical limit (85%). "
                       f"Reduce cable count or use a larger conduit.")
    return True, ""


def _circle_intersections(x0, y0, r0, x1, y1, r1):
    """
    Find the two intersection points of two circles.
    Returns a list of 0, 1, or 2 (x, y) tuples.
    Uses standard analytic geometry — no numpy required.
    """
    import math
    dx = x1 - x0
    dy = y1 - y0
    d  = math.sqrt(dx**2 + dy**2)

    # No intersection cases: circles separate, one inside other, or concentric
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0:
        return []

    a  = (r0**2 - r1**2 + d**2) / (2.0 * d)
    # Clamp to avoid domain errors from floating-point rounding
    h  = math.sqrt(max(0.0, r0**2 - a**2))

    # Midpoint along the line between centres
    mx = x0 + a * (dx / d)
    my = y0 + a * (dy / d)

    return [
        (mx + h * (dy / d), my - h * (dx / d)),
        (mx - h * (dy / d), my + h * (dx / d)),
    ]


def get_cable_positions(conduit_center_x, conduit_center_y,
                        conduit_radius_inner, jacket_radius, n_cables):
    """
    Sequential tangential placement — gravity-based cable settling.

    Key insight: all valid cable centres lie on a circle of radius
        R_track = conduit_radius_inner - jacket_radius
    centred at the conduit centre.  Finding where a new cable sits
    tangent to the wall AND tangent to an existing cable reduces to
    intersecting the R_track circle with a circle of radius
    2*jacket_radius around the existing cable centre.

    Placement order:
      1 — bottom centre (conduit floor)
      2 — rightmost intersection of R_track and spacing circle around #1
      3 — leftmost  intersection of R_track and spacing circle around #1
      4 — upper intersection of spacing circles around #2 and #3,
          verified to lie within the conduit

    Returns list of (cx, cy) tuples; may be shorter than n_cables if
    geometry constraints prevent placement.
    """
    if n_cables == 0:
        return []

    ok, _ = validate_cable_fit(conduit_radius_inner, jacket_radius, n_cables)
    if not ok:
        return []

    ccx, ccy = conduit_center_x, conduit_center_y
    R_track = conduit_radius_inner - jacket_radius

    if R_track <= 0:
        return []

    positions = []

    # ── Cable 1: bottom centre ──────────────────────────────────────────────
    c1 = (ccx, ccy - R_track)
    positions.append(c1)
    if n_cables == 1:
        return positions

    # ── Cable 2: right side, wall-tangent + tangent to cable 1 ─────────────
    # Intersect R_track circle (centre=conduit_centre) with
    # spacing circle (centre=c1, radius=2*jacket_radius).
    candidates = _circle_intersections(
        ccx, ccy, R_track,
        c1[0], c1[1], 2.0 * jacket_radius
    )
    c2 = max(candidates, key=lambda p: p[0]) if candidates else None
    if c2:
        positions.append(c2)
    if n_cables == 2:
        return positions

    # ── Cable 3: left side, same logic ─────────────────────────────────────
    candidates = _circle_intersections(
        ccx, ccy, R_track,
        c1[0], c1[1], 2.0 * jacket_radius
    )
    c3 = min(candidates, key=lambda p: p[0]) if candidates else None
    if c3:
        positions.append(c3)
    if n_cables == 3:
        return positions

    # ── Cable 4: valley between cables 2 and 3 ─────────────────────────────
    if c2 and c3:
        candidates = _circle_intersections(
            c2[0], c2[1], 2.0 * jacket_radius,
            c3[0], c3[1], 2.0 * jacket_radius
        )
        if candidates:
            # Take the upper point (higher y = sits in the valley above c2/c3)
            c4 = max(candidates, key=lambda p: p[1])
            # Verify it lies within the conduit inner wall
            import math
            dist = math.sqrt((c4[0] - ccx)**2 + (c4[1] - ccy)**2)
            if dist <= R_track + 1e-6:
                positions.append(c4)

    return positions

# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(conduit_configs, surrounding_k, T_ambient,
                   domain_w, domain_h, global_T_conductor):
    """
    conduit_configs: list of dicts, each with keys:
        center_x, center_y, OD_m, ID_m, conduit_k,
        cables: list of dicts {copper_r, jacket_r, conductor_k, jacket_k,
                               T_conductor, cable_label}
        conduit_label, deleted
    Returns dict of results.
    """
    active_conduits = [c for c in conduit_configs if not c.get("deleted", False)]

    # Build mesh
    res = 200
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [domain_w, domain_h]],
        [res, res]
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q_space = fem.functionspace(domain, ("DG", 0))
    kappa = fem.Function(Q_space)
    kappa.x.array[:] = surrounding_k

    def in_circle(x, cx, cy, r):
        return (x[0]-cx)**2 + (x[1]-cy)**2 < r**2

    # Assign conductivities: order matters (last assignment wins)
    # 1. Surrounding (already set)
    # 2. Conduit walls
    for cd in active_conduits:
        cx, cy = cd["center_x"], cd["center_y"]
        r_out = cd["OD_m"] / 2
        r_in  = cd["ID_m"] / 2
        def conduit_wall(x, cx=cx, cy=cy, r_out=r_out, r_in=r_in):
            return np.logical_and(in_circle(x,cx,cy,r_out), ~in_circle(x,cx,cy,r_in))
        cells_wall = mesh.locate_entities(domain, domain.topology.dim, conduit_wall)
        kappa.x.array[cells_wall] = cd["conduit_k"]

        # Air inside conduit
        def conduit_air(x, cx=cx, cy=cy, r_in=r_in,
                        cables=cd["cables"]):
            inside = in_circle(x, cx, cy, r_in)
            for cab in cables:
                inside = np.logical_and(inside, ~in_circle(x, cab["cx"], cab["cy"], cab["jacket_r"]))
            return inside
        cells_air = mesh.locate_entities(domain, domain.topology.dim, conduit_air)
        kappa.x.array[cells_air] = 0.026  # air

        # Jacket and copper for each cable
        for cab in cd["cables"]:
            bcx, bcy = cab["cx"], cab["cy"]
            jr = cab["jacket_r"]
            cr = cab["copper_r"]
            def jacket_region(x, bcx=bcx, bcy=bcy, jr=jr, cr=cr):
                return np.logical_and(in_circle(x,bcx,bcy,jr), ~in_circle(x,bcx,bcy,cr))
            def copper_region(x, bcx=bcx, bcy=bcy, cr=cr):
                return in_circle(x, bcx, bcy, cr)
            cells_jk = mesh.locate_entities(domain, domain.topology.dim, jacket_region)
            cells_cu = mesh.locate_entities(domain, domain.topology.dim, copper_region)
            kappa.x.array[cells_jk] = cab["jacket_k"]
            kappa.x.array[cells_cu] = cab["conductor_k"]

    # Boundary conditions
    f = fem.Constant(domain, PETSc.ScalarType(0.0))

    def outer_boundary(x):
        tol = 1e-10
        return np.logical_or(
            np.logical_or(np.isclose(x[0],0.0,atol=tol), np.isclose(x[0],domain_w,atol=tol)),
            np.logical_or(np.isclose(x[1],0.0,atol=tol), np.isclose(x[1],domain_h,atol=tol))
        )

    bcs = []
    outer_dofs = fem.locate_dofs_geometrical(V, outer_boundary)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(T_ambient), outer_dofs, V))

    # Per-cable conductor temperature BCs
    for cd in active_conduits:
        for cab in cd["cables"]:
            bcx, bcy, cr = cab["cx"], cab["cy"], cab["copper_r"]
            T_c = cab["T_conductor"]
            def cu_marker(x, bcx=bcx, bcy=bcy, cr=cr):
                return in_circle(x, bcx, bcy, cr)
            cu_dofs = fem.locate_dofs_geometrical(V, cu_marker)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(T_c), cu_dofs, V))

    # Solve
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.dot(ufl.grad(u), ufl.grad(ufl.conj(v))) * ufl.dx
    L = f * ufl.conj(v) * ufl.dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="thermal_v2"
    )
    uh = problem.solve()

    # Extract mesh data
    gdim = domain.geometry.dim
    pts = domain.geometry.x
    connectivity = domain.topology.connectivity(domain.topology.dim, 0)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cells_arr = np.array([connectivity.links(i) for i in range(num_cells)])
    points_2d = pts[:, :gdim]
    t_vals = uh.x.array.real

    # Compute conduit wall temperatures
    conduit_stats = []
    for cd in active_conduits:
        cx, cy = cd["center_x"], cd["center_y"]
        r_out = cd["OD_m"] / 2
        r_in  = cd["ID_m"] / 2
        dist = np.sqrt((points_2d[:,0]-cx)**2 + (points_2d[:,1]-cy)**2)
        wall_mask = (dist >= r_in) & (dist <= r_out)
        wall_temps = t_vals[wall_mask]
        cable_stats = []
        for cab in cd["cables"]:
            bcx, bcy, jr = cab["cx"], cab["cy"], cab["jacket_r"]
            jdist = np.sqrt((points_2d[:,0]-bcx)**2 + (points_2d[:,1]-bcy)**2)
            jacket_mask = jdist <= jr
            jtemps = t_vals[jacket_mask]
            cable_stats.append({
                "label": cab["cable_label"],
                "T_max": float(np.max(jtemps)) if len(jtemps) else np.nan,
                "T_min": float(np.min(jtemps)) if len(jtemps) else np.nan,
                "T_avg": float(np.mean(jtemps)) if len(jtemps) else np.nan,
                "T_conductor": cab["T_conductor"],
            })
        conduit_stats.append({
            "label": cd["conduit_label"],
            "T_max": float(np.max(wall_temps)) if len(wall_temps) else np.nan,
            "T_min": float(np.min(wall_temps)) if len(wall_temps) else np.nan,
            "T_avg": float(np.mean(wall_temps)) if len(wall_temps) else np.nan,
            "conduit_k": cd["conduit_k"],
            "max_temp_rating": cd.get("max_temp_rating", 60.0),
            "cable_stats": cable_stats,
        })

    return {
        "t_vals": t_vals,
        "points": points_2d,
        "cells": cells_arr,
        "conduit_stats": conduit_stats,
        "T_min": float(np.min(t_vals)),
        "T_max": float(np.max(t_vals)),
        "T_avg": float(np.mean(t_vals)),
        "active_conduits": active_conduits,
    }

# ============================================================================
# PLOTTING
# ============================================================================

def make_plots(results, domain_w, domain_h, T_ambient, T_max_global):
    t_vals   = results["t_vals"]
    points   = results["points"]
    cells    = results["cells"]
    conduits = results["active_conduits"]

    triang = tri.Triangulation(points[:,0], points[:,1], cells)

    # Calculate zoom bounds around all conduits
    if conduits:
        all_cx = [c["center_x"] for c in conduits]
        all_cy = [c["center_y"] for c in conduits]
        zoom_cx = (min(all_cx)+max(all_cx))/2
        zoom_cy = (min(all_cy)+max(all_cy))/2
        max_r_out = max(c["OD_m"]/2 for c in conduits)
        margin = max_r_out + 0.12
        half_span = max(
            (max(all_cx)-min(all_cx))/2 + margin,
            (max(all_cy)-min(all_cy))/2 + margin,
            0.18
        )
        x0 = max(0, zoom_cx - half_span)
        x1 = min(domain_w, zoom_cx + half_span)
        y0 = max(0, zoom_cy - half_span)
        y1 = min(domain_h, zoom_cy + half_span)
    else:
        x0, x1 = 0, domain_w
        y0, y1 = 0, domain_h

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.35)

    levels = np.linspace(T_ambient, T_max_global, 20)

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 2:
            cf = ax.tricontourf(triang, t_vals, levels=levels, cmap='inferno', extend='both')
            ax.tricontour(triang, t_vals, levels=levels, colors='white', linewidths=0.4, alpha=0.3)
            plt.colorbar(cf, ax=ax, label='Temperature (°C)')
            title = "Temperature Contours"
        else:
            tc = ax.tripcolor(triang, t_vals, cmap='inferno', shading='gouraud',
                              vmin=T_ambient, vmax=T_max_global)
            plt.colorbar(tc, ax=ax, label='Temperature (°C)')
            title = "Full Domain" if ax_idx == 0 else "Conduit Detail"

        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)

        if ax_idx == 0:
            ax.set_xlim(0, domain_w)
            ax.set_ylim(0, domain_h)
        else:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        # Draw conduit geometry
        for cd in conduits:
            cx, cy = cd["center_x"], cd["center_y"]
            r_out = cd["OD_m"] / 2
            r_in  = cd["ID_m"] / 2
            label = cd["conduit_label"]

            ax.add_patch(plt.Circle((cx,cy), r_out, fill=False,
                                    edgecolor='white', lw=2.0, linestyle='-'))
            if ax_idx != 0:
                ax.add_patch(plt.Circle((cx,cy), r_in, fill=False,
                                        edgecolor='white', lw=1.5, linestyle='--'))

            # Cable circles
            for cab in cd["cables"]:
                bcx, bcy = cab["cx"], cab["cy"]
                ax.add_patch(plt.Circle((bcx,bcy), cab["jacket_r"], fill=False,
                                        edgecolor='yellow', lw=1.5))
                ax.add_patch(plt.Circle((bcx,bcy), cab["copper_r"], fill=False,
                                        edgecolor='red', lw=1.2))
                if ax_idx == 0:
                    ax.text(bcx, bcy, cab["cable_label"],
                            ha='center', va='center',
                            fontsize=5, color='yellow',
                            fontweight='bold')

            # Conduit label top-left of conduit
            label_x = cx - r_out * 0.65
            label_y = cy + r_out * 0.75
            ax.text(label_x, label_y, label,
                    ha='center', va='center',
                    fontsize=7 if ax_idx == 0 else 9,
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='#333333', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Cable Thermal Analysis v2", layout="wide",
                       initial_sidebar_state="expanded")

    st.title("🔥 Cable Thermal Analysis Tool — v4")
    st.caption("Multi-conduit heat transfer analysis for buried cable installations")
    st.markdown("---")

    # ── Session state init ──────────────────────────────────────────────────
    if "deleted_conduits" not in st.session_state:
        st.session_state.deleted_conduits = set()
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    if "last_fig" not in st.session_state:
        st.session_state.last_fig = None

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Global Settings")

        # Preset loader
        preset_choice = st.selectbox("📋 Load Preset Scenario",
                                     ["(None)"] + list(PRESETS.keys()))
        preset = PRESETS.get(preset_choice, {})

        st.subheader("Conduit Grid")
        col_x, col_y = st.columns(2)
        with col_x:
            grid_x = st.number_input("Horizontal (#)",
                                     min_value=1, max_value=5,
                                     value=int(preset.get("grid_x", 1)), step=1)
        with col_y:
            grid_y = st.number_input("Vertical (#)",
                                     min_value=1, max_value=5,
                                     value=int(preset.get("grid_y", 1)), step=1)

        st.subheader("Conduit Spacing (center-to-center)")
        h_spacing_in = st.slider("Horizontal spacing (in)",
                                 min_value=3.0, max_value=24.0,
                                 value=float(preset.get("h_spacing", 6.0)),
                                 step=0.5)
        v_spacing_in = st.slider("Vertical spacing (in)",
                                 min_value=3.0, max_value=24.0,
                                 value=float(preset.get("v_spacing", 6.0)),
                                 step=0.5)

        h_spacing = h_spacing_in * 0.0254
        v_spacing = v_spacing_in * 0.0254

        st.subheader("Domain Size")
        # Auto minimum
        min_w = (grid_x - 1) * h_spacing + 0.40
        min_h = (grid_y - 1) * v_spacing + 0.40
        auto_w = max(1.0, round(min_w + 0.4, 1))
        auto_h = max(1.0, round(min_h + 0.4, 1))

        domain_w = st.slider("Domain Width (m)", 1.0, 5.0,
                             value=min(5.0, max(1.0, auto_w)),
                             step=0.1)
        domain_h = st.slider("Domain Height (m)", 1.0, 5.0,
                             value=min(5.0, max(1.0, auto_h)),
                             step=0.1)

        st.subheader("Environment")
        surrounding_name = st.selectbox("Surrounding Medium",
                                        list(SURROUNDING_MATERIALS.keys()),
                                        index=list(SURROUNDING_MATERIALS.keys()).index(
                                            preset.get("surrounding", "Moist Soil")))
        surrounding_k = SURROUNDING_MATERIALS[surrounding_name]["k"]

        st.subheader("Temperature")
        global_T_cond = st.slider("Global Conductor Temp (°C)",
                                  40, 105,
                                  value=int(preset.get("global_T_conductor", 70)),
                                  step=5)
        T_ambient = st.slider("Ambient/Soil Temp (°C)",
                              0, 50,
                              value=int(preset.get("T_ambient", 10)),
                              step=1)

        st.markdown("---")
        run_btn = st.button("🚀 Run Simulation", type="primary",
                            use_container_width=True)

    # ── Compute conduit center positions ────────────────────────────────────
    total_w = (grid_x - 1) * h_spacing
    total_h = (grid_y - 1) * v_spacing
    start_x = (domain_w - total_w) / 2
    start_y = (domain_h - total_h) / 2

    grid_positions = []   # list of (col, row, cx, cy, label_index)
    label_idx = 1
    for row in range(grid_y):
        for col in range(grid_x):
            cx = start_x + col * h_spacing
            cy = start_y + row * v_spacing
            grid_positions.append((col, row, cx, cy, label_idx))
            label_idx += 1

    # ── Boundary check ───────────────────────────────────────────────────────
    # Use default conduit OD for check (4 IPS SCH 40 HDPE = 4.5 in OD)
    default_r_out = (4.5 * 0.0254) / 2
    out_of_bounds = []
    for col, row, cx, cy, lidx in grid_positions:
        if (cx - default_r_out < 0 or cx + default_r_out > domain_w or
                cy - default_r_out < 0 or cy + default_r_out > domain_h):
            out_of_bounds.append(f"C{lidx}")

    if out_of_bounds:
        st.warning(
            f"⚠️ Conduit(s) {', '.join(out_of_bounds)} may exceed domain bounds. "
            f"Increase domain size or reduce spacing / grid size. "
            f"Simulation will proceed but results near boundary may be inaccurate."
        )

    # ── Per-conduit configuration ────────────────────────────────────────────
    st.subheader("🔧 Per-Conduit Configuration")
    st.caption("Configure each conduit individually. Use the delete button to remove a conduit from the simulation.")

    num_conduits = grid_x * grid_y
    # Reset deleted set if grid changes
    if f"grid_sig" not in st.session_state or st.session_state.grid_sig != (grid_x, grid_y):
        st.session_state.deleted_conduits = set()
        st.session_state.grid_sig = (grid_x, grid_y)

    conduit_configs_ui = {}   # label_idx -> config dict

    cols_per_row = min(grid_x, 4)
    for row_i in range(grid_y):
        ui_cols = st.columns(cols_per_row)
        for col_i in range(grid_x):
            pos_idx = row_i * grid_x + col_i
            col, row, cx, cy, lidx = grid_positions[pos_idx]
            ui_col = ui_cols[col_i % cols_per_row]
            with ui_col:
                deleted = lidx in st.session_state.deleted_conduits
                header_color = "~~" if deleted else ""
                with st.expander(
                    f"{'🚫 ' if deleted else ''}C{lidx}  "
                    f"({'Deleted' if deleted else f'({cx:.2f},{cy:.2f})m'})",
                    expanded=not deleted
                ):
                    if deleted:
                        st.caption("This conduit is deleted (surrounding medium fills this location).")
                        if st.button(f"↩️ Restore C{lidx}", key=f"restore_{lidx}"):
                            st.session_state.deleted_conduits.discard(lidx)
                            st.rerun()
                    else:
                        # Material & size
                        mat = st.selectbox("Material", ["HDPE", "PVC"],
                                           key=f"mat_{lidx}")
                        ips_opts = list(HDPE_DIMS.keys() if mat == "HDPE" else PVC_DIMS.keys())
                        ips = st.selectbox("Size (IPS)", ips_opts,
                                          index=1, key=f"ips_{lidx}")  # default 4 IPS
                        wall_opts = get_wall_types(mat, ips)
                        wall = st.selectbox("Wall Type", wall_opts,
                                           key=f"wall_{lidx}")
                        od_m, id_m = get_conduit_dims(mat, ips, wall)
                        st.caption(f"OD: {od_m/0.0254:.3f}\" | ID: {id_m/0.0254:.3f}\"")

                        # Cable config
                        n_cables = st.slider("# Cables", 0, 4, 1,
                                            key=f"ncab_{lidx}")
                        if n_cables > 0:
                            cable_size = st.selectbox(
                                "Cable Size",
                                list(CABLE_SIZES.keys()),
                                index=2,  # 750 kcmil default
                                key=f"csize_{lidx}"
                            )
                            # Validate fit immediately in UI
                            _cdata = CABLE_SIZES[cable_size]
                            _jr = (_cdata["jacket_OD"] * 0.0254) / 2
                            _fit_ok, _fit_msg = validate_cable_fit(id_m/2, _jr, n_cables)
                            if not _fit_ok:
                                st.error(f"⛔ {_fit_msg}")
                            elif n_cables > 1:
                                _fill = (n_cables * np.pi * _jr**2) / (np.pi * (id_m/2)**2)
                                st.caption(f"Fill ratio: {_fill*100:.0f}%")
                            cond_mat = st.selectbox(
                                "Conductor",
                                list(CABLE_CONDUCTOR_MATERIALS.keys()),
                                key=f"cmat_{lidx}"
                            )
                            jack_mat = st.selectbox(
                                "Jacket",
                                list(CABLE_JACKET_MATERIALS.keys()),
                                key=f"jmat_{lidx}"
                            )
                            override_temp = st.checkbox(
                                f"Override conductor temp",
                                key=f"ovr_{lidx}"
                            )
                            if override_temp:
                                T_cond_local = st.slider(
                                    "Conductor Temp (°C)",
                                    40, 105, global_T_cond, step=5,
                                    key=f"tcond_{lidx}"
                                )
                            else:
                                T_cond_local = global_T_cond
                        else:
                            cable_size = "750 kcmil"
                            cond_mat = "Copper"
                            jack_mat = "XLPE"
                            T_cond_local = global_T_cond

                        if st.button(f"🗑️ Delete C{lidx}", key=f"del_{lidx}",
                                    type="secondary"):
                            st.session_state.deleted_conduits.add(lidx)
                            st.rerun()

                        conduit_configs_ui[lidx] = {
                            "mat": mat, "ips": ips, "wall": wall,
                            "od_m": od_m, "id_m": id_m,
                            "n_cables": n_cables,
                            "cable_size": cable_size,
                            "cond_mat": cond_mat,
                            "jack_mat": jack_mat,
                            "T_cond_local": T_cond_local,
                        }

    # ── Build simulation config ──────────────────────────────────────────────
    conduit_configs = []
    for col, row, cx, cy, lidx in grid_positions:
        deleted = lidx in st.session_state.deleted_conduits
        if deleted:
            conduit_configs.append({"deleted": True, "conduit_label": f"C{lidx}"})
            continue

        ui = conduit_configs_ui.get(lidx, {})
        if not ui:
            continue

        od_m = ui["od_m"]
        id_m = ui["id_m"]
        r_in = id_m / 2

        cable_data = CABLE_SIZES[ui["cable_size"]]
        copper_r = (cable_data["copper_OD"] * 0.0254) / 2
        jacket_r = (cable_data["jacket_OD"] * 0.0254) / 2
        conductor_k = CABLE_CONDUCTOR_MATERIALS[ui["cond_mat"]]["k"]
        jacket_k = CABLE_JACKET_MATERIALS[ui["jack_mat"]]["k"]

        # Validate cable fit — silently drop cables that don't fit
        fit_ok, fit_msg = validate_cable_fit(r_in, jacket_r, ui["n_cables"])
        actual_n = ui["n_cables"] if fit_ok else 0

        cable_positions = get_cable_positions(cx, cy, r_in, jacket_r, actual_n)
        cables = []
        for b_idx, (bcx, bcy) in enumerate(cable_positions):
            cables.append({
                "cx": bcx, "cy": bcy,
                "copper_r": copper_r,
                "jacket_r": jacket_r,
                "conductor_k": conductor_k,
                "jacket_k": jacket_k,
                "T_conductor": ui["T_cond_local"],
                "cable_label": f"B{b_idx+1}",
            })

        conduit_configs.append({
            "center_x": cx, "center_y": cy,
            "OD_m": od_m, "ID_m": id_m,
            "conduit_k": CONDUIT_MATERIALS[ui["mat"]]["k"],
            "max_temp_rating": CONDUIT_MATERIALS[ui["mat"]]["max_temp"],
            "conduit_label": f"C{lidx}",
            "cables": cables,
            "deleted": False,
        })

    # ── Run simulation ────────────────────────────────────────────────────────
    if run_btn:
        active_count = sum(1 for c in conduit_configs if not c.get("deleted", False))
        if active_count == 0:
            st.error("❌ No active conduits to simulate. Restore at least one conduit.")
        else:
            with st.spinner("🔄 Running simulation... please wait."):
                try:
                    results = run_simulation(
                        conduit_configs, surrounding_k, T_ambient,
                        domain_w, domain_h, global_T_cond
                    )
                    T_max_all = max(global_T_cond, results["T_max"])
                    fig = make_plots(results, domain_w, domain_h,
                                     T_ambient, T_max_all)
                    st.session_state.last_results = results
                    st.session_state.last_fig = fig
                    st.success("✅ Simulation complete!")
                except Exception as e:
                    st.error(f"❌ Simulation error: {e}")
                    st.exception(e)

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        fig     = st.session_state.last_fig

        st.markdown("---")
        st.subheader("📊 Summary Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Domain Max Temp", f"{results['T_max']:.2f}°C")
        mc2.metric("Domain Min Temp", f"{results['T_min']:.2f}°C")
        mc3.metric("Domain Avg Temp", f"{results['T_avg']:.2f}°C")
        mc4.metric("Active Conduits", len(results['active_conduits']))

        st.subheader("🌡️ Temperature Distribution")
        st.pyplot(fig, use_container_width=True)

        # ── Detailed statistics ──────────────────────────────────────────────
        with st.expander("📈 Detailed Statistics by Conduit & Cable", expanded=True):
            for cs in results["conduit_stats"]:
                warn = (cs["T_max"] > cs["max_temp_rating"])
                header = f"{'⚠️ ' if warn else '✅ '}{cs['label']}"
                st.markdown(f"**{header}**")
                if warn:
                    st.warning(
                        f"⚠️ {cs['label']} wall max temp ({cs['T_max']:.1f}°C) "
                        f"exceeds material rating ({cs['max_temp_rating']:.0f}°C)!"
                    )
                c_col1, c_col2, c_col3 = st.columns(3)
                c_col1.metric(f"{cs['label']} Wall Max", f"{cs['T_max']:.2f}°C")
                c_col2.metric(f"{cs['label']} Wall Min", f"{cs['T_min']:.2f}°C")
                c_col3.metric(f"{cs['label']} Wall Avg", f"{cs['T_avg']:.2f}°C")

                if cs["cable_stats"]:
                    cable_rows = []
                    for cab in cs["cable_stats"]:
                        cable_rows.append({
                            "Cable": f"{cs['label']}-{cab['label']}",
                            "Conductor Temp (°C)": cab["T_conductor"],
                            "Max Temp (°C)": f"{cab['T_max']:.2f}",
                            "Min Temp (°C)": f"{cab['T_min']:.2f}",
                            "Avg Temp (°C)": f"{cab['T_avg']:.2f}",
                        })
                    st.dataframe(pd.DataFrame(cable_rows),
                                 use_container_width=True, hide_index=True)
                else:
                    st.caption("No cables in this conduit.")
                st.markdown("---")

        # ── Thermal resistance table ─────────────────────────────────────────
        with st.expander("🔬 Thermal Resistance Summary"):
            st.caption("Approximate thermal resistance per unit length (K·m/W) for each layer")
            r_rows = []
            for cs_cfg, cs_stat in zip(
                [c for c in conduit_configs if not c.get("deleted", False)],
                results["conduit_stats"]
            ):
                r_out = cs_cfg["OD_m"] / 2
                r_in  = cs_cfg["ID_m"] / 2
                k_c   = cs_cfg["conduit_k"]
                R_wall = np.log(r_out / r_in) / (2 * np.pi * k_c)
                r_rows.append({
                    "Conduit": cs_stat["label"],
                    "Layer": "Conduit Wall",
                    "R (K·m/W)": f"{R_wall:.4f}",
                    "k (W/m·K)": f"{k_c:.3f}",
                })
                for cab_cfg, cab_stat in zip(cs_cfg["cables"], cs_stat["cable_stats"]):
                    jr = cab_cfg["jacket_r"]
                    cr = cab_cfg["copper_r"]
                    kj = cab_cfg["jacket_k"]
                    R_jacket = np.log(jr / cr) / (2 * np.pi * kj)
                    r_rows.append({
                        "Conduit": cs_stat["label"],
                        "Layer": f"{cab_stat['label']} Jacket",
                        "R (K·m/W)": f"{R_jacket:.4f}",
                        "k (W/m·K)": f"{kj:.3f}",
                    })
            if r_rows:
                st.dataframe(pd.DataFrame(r_rows),
                             use_container_width=True, hide_index=True)

        # ── Downloads ────────────────────────────────────────────────────────
        st.subheader("💾 Download Results")
        dl1, dl2 = st.columns(2)

        with dl1:
            png_buf = BytesIO()
            fig.savefig(png_buf, format='png', dpi=200, bbox_inches='tight')
            png_buf.seek(0)
            st.download_button("📥 Download Plot (PNG)", data=png_buf,
                               file_name="thermal_analysis_v2.png",
                               mime="image/png")

        with dl2:
            report_rows = []
            for cs in results["conduit_stats"]:
                report_rows.append({
                    "Component": cs["label"],
                    "Type": "Conduit Wall",
                    "Max Temp (°C)": round(cs["T_max"], 2),
                    "Min Temp (°C)": round(cs["T_min"], 2),
                    "Avg Temp (°C)": round(cs["T_avg"], 2),
                    "Temp Rating (°C)": cs["max_temp_rating"],
                    "Rating Exceeded": cs["T_max"] > cs["max_temp_rating"],
                })
                for cab in cs["cable_stats"]:
                    report_rows.append({
                        "Component": f"{cs['label']}-{cab['label']}",
                        "Type": "Cable",
                        "Max Temp (°C)": round(cab["T_max"], 2),
                        "Min Temp (°C)": round(cab["T_min"], 2),
                        "Avg Temp (°C)": round(cab["T_avg"], 2),
                        "Temp Rating (°C)": "",
                        "Rating Exceeded": "",
                    })
            csv_buf = BytesIO()
            pd.DataFrame(report_rows).to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button("📥 Download Report (CSV)", data=csv_buf,
                               file_name="thermal_analysis_v2.csv",
                               mime="text/csv")

    else:
        st.info("👈 Configure parameters in the sidebar and per-conduit settings above, then click **Run Simulation**.")
        if preset_choice != "(None)":
            st.success(f"📋 Preset loaded: **{preset_choice}** — {PRESETS[preset_choice]['description']}")

if __name__ == "__main__":
    main()
