# iiFBA: Iterative Interaction Flux Balance Analysis

## Project Overview

iiFBA is a Python package that extends COBRApy to simulate iterative interactions between microbial communities using flux balance analysis. The core algorithm models metabolic cross-feeding by iteratively updating environmental fluxes based on organism exchange reactions until convergence.

## Architecture & Components

### Core Module Structure
- **`Community`** class (`community.py`): Modern object-oriented interface - use this for new code
- **`analysis.py`**: Legacy functional interface with `iipfba()` and `iisampling()` functions
- **`utils.py`**: Input validation, visualization (`iifba_vis`), and example model loaders
- **`summary.py`**: `CommunitySummary` class for analyzing results with rich HTML/text output
- **`config.py`**: Global constants (`GROWTH_MIN_OBJ`, `ROUND`, `V_MAX`)

### Data Flow Pattern
1. **Initialization**: Environment fluxes from media, organism flux storage with multi-indexing
2. **Iteration Loop**: Set exchange bounds → Run FBA/pFBA → Update environment → Check convergence
3. **Output**: Multi-indexed DataFrames for environment and organism fluxes by iteration

## Key Development Patterns

### Input Validation & Type Handling
All functions use strict input validation via `utils.check_*()` functions:
```python
models = utils.check_models(models)  # Always returns list, even for single model
rel_abund = utils.check_rel_abund(rel_abund, size)  # Auto-normalizes to sum=1
media = utils.check_media(media)  # Dict: {"EX_glc(e)": -10}
```

### DataFrame Structure Convention
- **Environment fluxes**: Multi-indexed by `(Iteration, Run)`, columns are exchange reaction IDs
- **Organism fluxes**: Multi-indexed by `(Model, Iteration, Run)`, columns are all reaction IDs
- **Index naming**: Always use `["Iteration", "Run"]` and `["Model", "Iteration", "Run"]`

### Exchange Reaction Mapping
Critical pattern for linking exchange reactions to metabolites:
```python
self.ex_to_met = {}  # Maps "EX_glc(e)" -> "glc_e"
for rxn in model.exchanges:
    mets = list(rxn.metabolites.keys())
    if len(mets) == 1:
        self.ex_to_met[rxn.id] = mets[0].id
```

### Convergence & Iteration Control
- **Early stopping**: Compare environment fluxes between iterations with `1e-6` tolerance
- **Over-saturation handling**: When organisms demand more than available, scale down proportionally
- **Flux bounds**: Always negative for uptake bounds: `ex.lower_bound = -flux_value`

## Development Workflows

### Running Examples
```bash
# Install package in development mode
cd package && pip install -e .

# Load examples
from iifba.utils import load_example_models, load_simple_models
models, media = load_simple_models(3)  # Numbers 1-7 for different scenarios
```

### Testing with Simple Models
Use `load_simple_models(1-7)` for validation:
- Models 1-2: Single organism scenarios
- Models 3-7: Two-organism interaction scenarios with cross-feeding
- Located in `package/iifba/Simple_Models/` as JSON files

### Debugging Patterns
- Set `v=True` for verbose iteration output
- Check `community.env_fluxes` and `community.org_fluxes` DataFrames directly
- Use `summary()` method for formatted results analysis

## Integration Points

### COBRApy Integration
- Models must be `cobra.Model` objects with proper exchange reactions
- Uses `model.slim_optimize()` for growth checks (threshold: `GROWTH_MIN_OBJ = 0.01`)
- Supports both FBA and parsimonious FBA (pFBA) objectives
- Context managers (`with model as m:`) for temporary modifications

### Data Analysis Integration
- All outputs are pandas DataFrames for easy analysis
- `iifba_vis()` creates matplotlib plots of cumulative fluxes
- `CommunitySummary` provides rich output for Jupyter notebooks

## Common Pitfalls
- Always pass models as lists, even single models: `[model]` not `model`
- Media fluxes must be negative for uptake: `{"EX_glc(e)": -10}`
- Don't modify original models - algorithm uses context managers
- Relative abundances auto-normalize but must sum > 0
