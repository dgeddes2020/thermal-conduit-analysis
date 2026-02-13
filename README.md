# Cable Thermal Analysis Tool

A finite element analysis (FEA) web application for modelling steady-state heat transfer from buried electrical cable installations. Simulates heat conduction from cable conductors through cable jackets, air gaps, conduit walls, and into surrounding soil or concrete — all in a 2D cross-sectional view.

Built with [FEniCS/DOLFINx](https://fenicsproject.org/) for the physics solver and [Streamlit](https://streamlit.io/) for the web interface.

---

## Table of Contents

- [What This Tool Does](#what-this-tool-does)
- [How the FEA Works](#how-the-fea-works)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Installation and Setup](#installation-and-setup)
- [Running the App](#running-the-app)
- [How to Use the Interface](#how-to-use-the-interface)
- [Material Data Sources and Standards](#material-data-sources-and-standards)
- [Cable Positioning Logic](#cable-positioning-logic)
- [Project Structure](#project-structure)

---

## What This Tool Does

Given a grid of buried conduits (each containing one or more power cables), this tool:

1. Builds a 2D finite element mesh of the cross-section
2. Assigns thermal conductivity values to each material region (copper, jacket, air, conduit wall, soil/concrete)
3. Applies temperature boundary conditions (conductor surface temperature, far-field soil temperature)
4. Solves the steady-state heat conduction equation
5. Reports temperature distributions at conduit walls and cable surfaces
6. Flags if any conduit wall temperature exceeds the material rated operating temperature

---

## How the FEA Works

### Governing Equation

The tool solves the **steady-state heat conduction equation** (Poisson equation):

```
-∇ · (κ ∇T) = Q    in Ω
```

Where:
- `T` is temperature (degrees C)
- `κ` is the spatially-varying thermal conductivity (W/m·K)
- `Q` is the volumetric heat source (set to zero here — conductor temperature is applied as a Dirichlet boundary condition instead)
- `Ω` is the 2D cross-sectional domain

### Finite Element Method

DOLFINx uses the **Galerkin finite element method** to discretise this equation:

**1. Mesh generation**
The domain is divided into a structured triangular mesh (200×200 elements by default). Each element is assigned a single thermal conductivity value based on which material region it falls in, using a discontinuous Galerkin DG0 function space. This allows sharp conductivity transitions at material interfaces (e.g. conduit wall to air, air to cable jacket).

**2. Weak form**
The PDE is converted to its integral (weak) form by multiplying by a test function `v` and integrating over the domain. For the complex-valued DOLFINx build, test functions are conjugated per the sesquilinear form convention:

```
a(u, v) = ∫ κ ∇u · ∇conj(v) dΩ = L(v) = ∫ Q · conj(v) dΩ
```

**3. Boundary conditions**
Two Dirichlet (fixed-value) boundary conditions are applied:
- **Conductor surface**: Fixed at the user-specified conductor operating temperature
- **Domain outer boundary**: Fixed at the user-specified ambient/soil temperature

These two conditions together drive the heat flow from hot conductor to cool soil, with the finite element method computing the temperature field everywhere in between.

**4. Linear solver**
The assembled linear system is solved using a direct LU factorisation via PETSc (using the MUMPS backend where available). Direct solvers are appropriate here because the problem is small-to-medium sized, symmetric positive definite, and requires an exact solution rather than an iterative approximation.

**5. Post-processing**
Nodal temperature values are extracted and mapped onto the triangular mesh for visualisation using Matplotlib `tripcolor` (smooth interpolated colour map) and `tricontourf` (filled contour lines). Per-conduit and per-cable statistics are computed by querying nodal temperatures within each geometric region.

### Why Steady-State?

This tool models the **steady-state** (time-independent) condition — the temperature field after the cable has been energised long enough for the thermal gradient to fully stabilise. For buried cable installations this typically takes hours to days, depending on soil thermal diffusivity. Steady-state analysis gives the worst-case continuous operating temperature, which is the relevant design condition for cable ampacity calculations per IEC 60287 and IEEE Std 835.

---

## Assumptions and Limitations

| Assumption | Details |
|---|---|
| 2D cross-section | The model is a per-unit-length slice. It does not account for axial heat flow along the cable route or end effects. |
| Steady-state only | No transient analysis. Results represent long-term continuous loading at constant current. |
| Isotropic conductivity | All materials conduct heat equally in all directions. Soil anisotropy is not modelled. |
| No convection in air gap | Air inside the conduit is modelled as a solid with low conductivity (0.026 W/m·K). Natural convection within the air space is neglected. This is slightly conservative and will over-predict the temperature across the air gap. |
| Uniform surrounding medium | The soil or concrete has a single uniform conductivity. Layered soils or moisture migration are not modelled. |
| Fixed boundary temperature | Far-field temperature at the domain edge is fixed. The domain should be large enough that boundary heating from the cable is negligible — typically 1 to 3 m radius is sufficient for single conduit installations. |
| Conductor as isothermal surface | The copper conductor is a fixed-temperature boundary, not a volumetric heat source. This is appropriate when the conductor operating temperature is known. To model heat generation from current loading, a volumetric source term Q = I²ρ/A could be substituted. |
| No soil dry-out | Soil thermal dry-out around hot cables, which can significantly reduce soil conductivity, is not modelled. |
| Cable fill geometry | Multiple cables are positioned using a simplified gravity-based tangency algorithm. Real cable lay may differ. |

---

## Installation and Setup

### Prerequisites

This tool requires FEniCS/DOLFINx, which is most reliably installed via Docker. A standard pip install of DOLFINx is not supported because it requires compiled C++ bindings for PETSc, MPI, and FFCX.

### Step 1 — Install Docker

Download and install Docker Desktop from https://www.docker.com/products/docker-desktop

### Step 2 — Pull the DOLFINx Docker Image

```bash
docker pull dolfinx/dolfinx:v0.9.0
```

This tool was developed and tested with the complex-valued scalar build of DOLFINx (dolfinx-complex). The default dolfinx/dolfinx image includes both real and complex builds. When using the complex build, the weak forms use conjugated test functions as shown in the code.

### Step 3 — Clone This Repository

```bash
git clone https://github.com/your-username/cable-thermal-analysis.git
cd cable-thermal-analysis
```

### Step 4 — Start the Docker Container

Mount the repository directory into the container and expose the Streamlit port:

```bash
docker run -it --rm \
  -v $(pwd):/home/user/app \
  -p 8501:8501 \
  dolfinx/dolfinx:v0.9.0 \
  /bin/bash
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`.

### Step 5 — Install Python Dependencies Inside the Container

```bash
cd /home/user/app
pip install -r requirements.txt
```

DOLFINx, UFL, MPI4Py, and PETSc4Py are already present in the Docker image and should not be reinstalled via pip — doing so may break the compiled bindings.

---

## Running the App

Inside the Docker container:

```bash
cd /home/user/app
streamlit run thermal_app_v3.py --server.address 0.0.0.0 --server.port 8501
```

Open your browser and navigate to:

```
http://localhost:8501
```

If you are running inside a JupyterLab environment (e.g. the DOLFINx JupyterLab Docker image), open a terminal from JupyterLab and run the same command. The app will be accessible at the same localhost address.

---

## How to Use the Interface

### 1. Load a Preset (Optional)

Use the Load Preset Scenario dropdown to pre-populate a common configuration:
- **Single Cable Baseline** — one 4" HDPE conduit, 750 kcmil copper, moist soil
- **Typical 2x2 Install** — four conduits in a 2x2 grid, 6 inch spacing
- **Concrete Encased Bank** — 3x2 bank in concrete encasement

### 2. Configure the Conduit Grid

Set the grid size (horizontal count by vertical count) and independent center-to-center spacing sliders for horizontal and vertical directions. The domain size sliders will auto-suggest a minimum size based on the grid layout, but you can increase it to model a greater extent of surrounding soil.

An automatic boundary check will warn you if any conduit would extend outside the domain.

### 3. Set Global Temperatures

- **Global Conductor Temp**: Default operating temperature for all cables (range 40 to 105°C). 90°C is a common XLPE rated temperature; 105°C represents worst-case overload.
- **Ambient/Soil Temp**: Far-field ground temperature at burial depth (typically 10 to 15°C in Canada at 1 m depth).

### 4. Configure Each Conduit

Each conduit in the grid gets its own configuration panel:
- **Material**: HDPE or PVC. Wall type options are filtered to the relevant standard for each material.
- **Size and Wall Type**: IPS nominal size with available wall types. Actual OD and ID are shown in inches.
- **Cable Count**: 0 to 4 cables per conduit. Set to 0 for an empty conduit (air-filled).
- **Cable Size, Conductor, Jacket**: Per-conduit cable specification.
- **Temperature Override**: Check to set a different conductor temperature for cables in this conduit, independent of the global slider.
- **Delete Button**: Removes the conduit from the simulation entirely. The grid location is replaced by surrounding medium. Use Restore to add it back.

### 5. Run Simulation

Click Run Simulation. Expected computation times:
- 1 to 4 conduits: approximately 5 to 20 seconds
- 6 to 9 conduits: approximately 20 to 60 seconds

### 6. Review Results

**Summary Metrics** — domain-wide max, min, and average temperatures plus active conduit count.

**Temperature Distribution Plots** — three side-by-side plots:
- Full domain view with conduit labels (C1, C2...) and cable labels (B1, B2...)
- Zoomed conduit detail showing both OD and ID rings, jacket, and conductor circles
- Temperature contour map (zoomed to conduit region)

**Detailed Statistics** — per-conduit wall temperature table (min, avg, max) with a warning badge if the wall temperature exceeds the material rated temperature. Per-cable temperature table showing each cable identified as C1-B1, C1-B2, etc.

**Thermal Resistance Table** — calculated R-values in K·m/W for conduit walls and cable jackets using the standard cylindrical shell formula R = ln(r_out/r_in) / (2π k).

**Downloads** — PNG plot at 200 DPI and CSV report with all temperature results and rating-exceeded flags.

---

## Material Data Sources and Standards

### Conduit Dimensions

| Material | Standard | Wall Types |
|---|---|---|
| HDPE | ASTM F2160 — Solid Wall HDPE Conduit | SCH 40, SCH 80, SDR 11, SDR 13.5 |
| PVC | ASTM D2665 / NEMA TC-2 — PVC Drain and Electrical Conduit | SCH 40, SCH 80 |

Dimensions are nominal values for IPS (Iron Pipe Size) nominal pipe sizes: 3", 4", 5", 6", and 8".

### Conduit Temperature Ratings

| Material | Max Temp | Reference |
|---|---|---|
| HDPE | 82°C | Plastics Pipe Institute Technical Note TN-11: Suggested Temperature Limits for Thermoplastic Piping and Conduit in Non-Pressure Applications |
| PVC | 60°C | ASTM D2665 / NEMA TC-2 |
| Steel | 300°C | ASTM A53 |
| Fiberglass | 120°C | ASTM D2996 |

### Thermal Conductivity Values

| Material | k (W/m·K) | Source |
|---|---|---|
| HDPE conduit | 0.48 | Literature / manufacturer data |
| PVC conduit | 0.19 | Literature / manufacturer data |
| Copper conductor | 385.0 | Standard reference value |
| Aluminum conductor | 205.0 | Standard reference value |
| XLPE jacket | 0.286 | IEC 60287-1-1 |
| EPR jacket | 0.250 | IEC 60287-1-1 |
| PVC jacket | 0.190 | IEC 60287-1-1 |
| PE jacket | 0.380 | IEC 60287-1-1 |
| Air (conduit fill) | 0.026 | Standard reference value |
| Dry Soil | 0.25 | IEEE Std 442 / IEC 60287-3-1 |
| Moist Soil | 1.00 | IEEE Std 442 / IEC 60287-3-1 |
| Wet Soil | 1.50 | IEEE Std 442 / IEC 60287-3-1 |
| Sand | 0.58 | IEEE Std 442 |
| Clay | 1.28 | IEEE Std 442 |
| Concrete | 1.40 | ACI 122R / IEEE Std 442 |

### Cable Dimensions

Cable outer diameter values (copper OD and jacket OD) are based on typical manufacturer data for XLPE-insulated copper power cables per **ICEA S-94-649** and **AEIC CS9** standards. Values are nominal and may vary between manufacturers.

---

## Cable Positioning Logic

When multiple cables are placed in a single conduit, the tool uses sequential tangential placement to simulate gravity-based settling. The algorithm positions each cable at the lowest geometrically valid location touching the conduit wall and previously placed cables.

**Placement order:**
1. Cable 1 — bottom centre, resting on the conduit inner floor
2. Cable 2 — to the right, tangent to conduit wall and Cable 1
3. Cable 3 — to the left, tangent to conduit wall and Cable 1
4. Cable 4 — in the valley above Cables 2 and 3, tangent to both (the highest, most stable position without touching the wall)

The math uses circle-circle tangency (finding the intersection of two offset circles) to locate the exact centre of each new cable. The conduit inner wall is treated as a large circle whose effective radius from a cable-tangency perspective is `R_inner - r_jacket`.

**Validation:** Before placing cables, the tool checks:
- That the cable jacket diameter is smaller than the conduit inner diameter
- That the total cable fill area does not exceed 85% of the conduit inner area

If either check fails, an error is displayed in the conduit panel and no cables are placed for that conduit.

---

## Project Structure

```
cable-thermal-analysis/
├── thermal_app_v4.py     # Main application (current version, fixed cable placement)
├── thermal_app_v3.py     # Main application (revision)
├── thermal_app_v2.py     # Previous version (reference)
├── thermal_app.py        # Original proof-of-concept
├── requirements.txt      # Python dependencies for Docker environment
└── README.md             # This file
```

---

## Potential Future Enhancements

- Transient (time-dependent) analysis for load cycling and fault conditions
- Volumetric heat source from current loading (I²R) rather than fixed conductor temperature
- Ampacity back-calculation: given a max conductor temperature, find the maximum current
- Non-uniform or layered soil profiles
- PDF report export with input summary and result tables
- Side-by-side scenario comparison mode
- Import/export of configuration as JSON

---

## License

MIT License. See LICENSE file for details.

---

## Acknowledgements

- FEniCS Project / DOLFINx — finite element solver framework
- Streamlit — web application framework
- Plastics Pipe Institute — TN-11 temperature rating guidance for HDPE
- IEEE Std 442 — Guide for Soil Thermal Resistivity Measurements
- IEC 60287 — Electric Cables: Calculation of the Current Rating
- Plastics Pipe Institute — HDPE conduit dimension standards
