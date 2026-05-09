# Dialysis Unit Scheduling Pipeline - Architecture Documentation

## Executive Summary

This document provides a comprehensive architectural overview of the Discrete Event Simulation (DES) system for evaluating patient scheduling heuristics in a resource-constrained dialysis unit. The system compares **Fixed Assignment** protocols against dynamic **First-In-First-Out (FIFO)** queuing methodologies using Monte Carlo methods and Paired Difference Testing.

---

## System Overview

### Four-Tier Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Module 1   │────▶│   Module 2   │────▶│   Module 3   │                │
│  │   Scenario   │     │   Strategy   │     │   Monte Carlo│                │
│  │   Creator    │     │   Processor  │     │   Batcher    │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│         │                   │                   │                           │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ShiftScenario        ShiftStatistics     List[ShiftStatistics]            │
│                                                                              │
│                              │                                                │
│                              ▼                                                │
│                       ┌──────────────┐                                       │
│                       │   Module 4   │                                       │
│                       │  Visualizer  │                                       │
│                       └──────────────┘                                       │
│                              │                                                │
│                              ▼                                                │
│                    Visualization Outputs                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### Module 1: Stochastic Scenario Creator

**File:** `src/scenario_generator.py`

**Purpose:** Generates deterministic baseline scenarios for a single operational shift by isolating environmental constraints from scheduling outcomes.

#### Key Responsibilities:
- Creates immutable `ShiftScenario` objects
- Executes binary defect probability check at T=0
- Removes defective machines from active resource pool
- Uses pseudo-random number generators with explicit seeds for reproducibility

#### Input:
- `SimulationConfig`: Configuration parameters including:
  - Static bounds (shift length: 300m, session duration: 240m)
  - Stochastic ranges (patient volume: 3-5, nurses: 2-4)
  - Probability metrics (machine defect: 15%)

#### Output:
```python
@dataclass(frozen=True)
class ShiftScenario:
    patient_arrivals: List[Dict[str, int]]  # [{id, arrival_min, setup_min}, ...]
    nurse_count: int
    machine_ready_times: Dict[int, int]      # {machine_id: ready_minute}
    defective_machine_ids: List[int]         # [machine_id, ...]
    scenario_seed: int                       # Reproducibility seed
```

#### Key Design Principle:
**Separation of Concerns:** Scenario generation is completely decoupled from strategy execution, enabling paired-difference testing where multiple strategies process identical scenarios.

---

### Module 2: Polymorphic Strategy Processor

**Files:** 
- `src/strategies/base_strategy.py` (Abstract contract)
- `src/strategies/fixed_strategy.py` (Fixed Assignment implementation)
- `src/strategies/fifo_strategy.py` (FIFO implementation)
- `src/strategies/utils/__init__.py` (Shared utilities)

#### Architectural Contract

All strategies must implement the `SchedulingStrategy` abstract base class:

```python
class SchedulingStrategy(ABC):
    @abstractmethod
    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

#### Bipartite Resource Constraint

The core simulation models a **bipartite dependency**: patients require BOTH a machine AND a nurse simultaneously to begin treatment.

```
Patient Journey:
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐    ┌──────────┐
│   Arrival   │───▶│ Wait for Machine +   │───▶│ Setup Phase     │───▶│ Dialysis │
│             │    │ Nurse (Bipartite)    │    │ (Nurse + Machine)│    │ Session  │
└─────────────┘    └──────────────────────┘    └─────────────────┘    └──────────┘
                         │                            │                      │
                         │                            │                      │
                         ▼                            ▼                      ▼
                    Queue Time                   ~10 minutes            240 minutes
                                                  │                      │
                                                  ▼                      ▼
                                             Nurse Released        Machine Locked
                                                                   (+ 60m cooldown)
```

#### Strategy Implementations:

**1. Fixed Assignment Strategy**
- Patient i is assigned to Machine i
- Patient waits only for their designated machine
- Defective machines cause extreme wait times (999,999 minutes penalty)

**2. FIFO Strategy**
- Patients processed in arrival time order
- Each patient takes first available machine-nurse pair globally
- More adaptive to resource constraints

#### Shared Utilities (`src/strategies/utils/`)

The utils module contains all helper classes and functions:

**State Tracking Classes:**
- `MachineState`: Tracks machine availability and assignments
- `NurseState`: Tracks nurse busy/free status
- `PatientSession`: Tracks patient timing throughout the shift

**Resource Management Functions:**
- `initialize_machine_states()`: Create machine states from scenario
- `initialize_nurse_states()`: Create nurse states
- `get_available_nurse()`: Find available nurse at given time
- `find_earliest_nurse_availability()`: Next nurse free time

**Timing Calculations:**
- `calculate_session_end()`: Compute session end time
- `calculate_machine_busy_until()`: Compute machine release time
- `calculate_wait_time()`: Compute patient wait duration
- `calculate_overrun()`: Compute shift overrun

**Utilization Functions:**
- `calculate_nurse_utilization()`: Nurse utilization fraction
- `calculate_machine_utilization()`: Machine utilization fraction
- `calculate_max_time()`: Maximum session end time

**Statistics Aggregation:**
- `aggregate_wait_statistics()`: Mean and max wait times
- `aggregate_overrun_statistics()`: Total shift overrun

**Constants:**
- `SESSION_DURATION_MINUTES = 240`
- `COOLDOWN_DURATION_MINUTES = 60`
- `SHIFT_END_MINUTES = 300`
- `EXTREME_WAIT_PENALTY = 999999.0`

#### Output Schema (Schema S):

```python
@dataclass(frozen=True)
class ShiftStatistics:
    strategy_name: str
    total_patients_processed: int
    mean_wait_time_minutes: float
    max_wait_time_minutes: float
    nurse_utilization_percent: float
    machine_utilization_percent: float
    shift_overrun_minutes: int
```

---

### Module 3: Monte Carlo Batcher

**File:** `src/monte_carlo_batcher.py`

**Purpose:** Orchestrates paired execution to achieve statistical significance through Monte Carlo simulation.

#### Key Design: Paired Difference Testing

For each iteration:
1. Generate ONE scenario with unique seed
2. Run ALL strategies against that SAME scenario
3. Collect statistics from each run

This ensures variance in outcomes is attributable to **strategy differences**, not stochastic variation.

#### Implementation:

```python
class MonteCarloBatcher:
    def __init__(
        self,
        config: SimulationConfig,
        strategies: List[SchedulingStrategy],
        n_iterations: int,
        global_seed: int = 0,
    )
    
    def run(self) -> List[ShiftStatistics]:
        # Returns n_iterations × n_strategies results
```

#### Output:
- `List[ShiftStatistics]`: Flat list containing results from all strategy-scenario pairs
- Ready for Pandas ingestion and statistical analysis

---

### Module 4: Analytical Visualizer

**File:** `src/visualizer.py`

**Purpose:** Transforms Monte Carlo outputs into visual insights operating strictly against Schema S.

#### Visualization Types:

1. **Wait Distribution Boxplots**
   - Shows distribution of `mean_wait_time_minutes` and `max_wait_time_minutes`
   - Highlights outliers and variance between strategies

2. **Resource Utilization Bar Charts**
   - Compares `nurse_utilization_percent` vs `machine_utilization_percent`
   - Identifies operational bottlenecks

3. **Overrun Histograms**
   - Distribution of `shift_overrun_minutes`
   - Quantifies auxiliary temporal load on facility operations

4. **Paired Difference Plot**
   - Mean difference with 95% confidence intervals
   - Statistical significance visualization

#### Data Flow:
```
List[ShiftStatistics] ──▶ Pandas DataFrame ──▶ Matplotlib/Seaborn Plots
```

---

## Configuration Management

**File:** `src/config.py`

The `SimulationConfig` dataclass centralizes all simulation parameters:

```python
@dataclass(frozen=True)
class SimulationConfig:
    # Static temporal bounds
    shift_duration_minutes: int = 300
    session_duration_minutes: int = 240
    machine_cooldown_minutes: int = 60
    
    # Stochastic resource ranges
    patient_volume: IntRange = IntRange(3, 5)
    nurse_count: IntRange = IntRange(2, 4)
    total_machines: int = 5
    
    # Stochastic event generators
    arrival_minute_sampler: Callable[[random.Random], int]
    setup_duration_minutes_sampler: Callable[[random.Random], int]
    
    # Probability metrics
    machine_defect_probability: float = 0.15
```

### Separation from Strategies

**Critical Architecture Decision:** Configuration is completely separated from strategy logic:

1. **Config owns all parameters** - Strategies have NO hardcoded values
2. **Strategies import constants from utils** - Not from config directly
3. **Scenario generator bridges config and strategies** - Converts config to scenario

This enables:
- Easy parameter sweeps without modifying strategies
- Clear testing boundaries
- Reproducible experiments via seed control

---

## Data Models

### Core Dataclasses

Located in `src/models.py`:

```python
@dataclass(frozen=True, slots=True)
class ShiftScenario:
    """Immutable snapshot of shift initial conditions."""
    patient_arrivals: List[Dict[str, int]]
    nurse_count: int
    machine_ready_times: Dict[int, int]
    defective_machine_ids: List[int]
    scenario_seed: int


@dataclass(frozen=True, slots=True)
class ShiftStatistics:
    """Performance metrics from processing a shift."""
    strategy_name: str
    total_patients_processed: int
    mean_wait_time_minutes: float
    max_wait_time_minutes: float
    nurse_utilization_percent: float
    machine_utilization_percent: float
    shift_overrun_minutes: int
```

### Design Principles:
- **Immutability:** Both classes are frozen to prevent accidental modification
- **Slots:** Memory optimization for large-scale Monte Carlo runs
- **Clear Contracts:** Well-defined interfaces between modules

---

## Execution Flow

### Complete Pipeline Example:

```python
# 1. Initialize configuration
config = SimulationConfig()

# 2. Create strategies
strategies = [FixedStrategy(), FIFOStrategy()]

# 3. Initialize Monte Carlo batcher
batcher = MonteCarloBatcher(
    config=config,
    strategies=strategies,
    n_iterations=100,
    global_seed=42
)

# 4. Run simulation (paired execution)
results = batcher.run()  # Returns 200 ShiftStatistics objects

# 5. Generate visualizations
viz = Visualizer()
fig = viz.plot_wait_distribution(results)
fig.savefig("outputs/wait_distribution.png")
```

---

## Key Architectural Decisions

### 1. Decoupled Scenario Generation
Scenarios are generated independently of strategies, enabling:
- **Paired-difference testing:** Same scenario → different strategies
- **Statistical validity:** Variance attributed to algorithms, not randomness
- **Reproducibility:** Seed-based scenario recreation

### 2. Polymorphic Strategy Interface
Abstract base class ensures:
- **Interchangeability:** Any strategy can be swapped in
- **Extensibility:** New strategies require no pipeline changes
- **Type safety:** IDE and type checker support

### 3. Bipartite Resource Modeling
Explicit modeling of machine+nurse dependency:
- **Realistic bottlenecks:** Prevents oversimplified capacity assumptions
- **Authentic queue dynamics:** Captures real-world constraints

### 4. Utility Module for Strategies
Centralized helper functions provide:
- **Code reuse:** Common logic written once
- **Consistency:** All strategies use same calculations
- **Maintainability:** Changes propagate automatically

### 5. Immutable Data Structures
Frozen dataclasses ensure:
- **Thread safety:** Safe for parallel execution
- **Debugging clarity:** No unexpected mutations
- **Contract enforcement:** Clear data flow direction

---

## Directory Structure

```
/workspace/
├── main.py                          # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py                    # Simulation configuration
│   ├── models.py                    # Core dataclasses
│   ├── scenario_generator.py        # Module 1
│   ├── monte_carlo_batcher.py       # Module 3
│   ├── statistics_aggregator.py     # Statistical analysis
│   ├── visualizer.py                # Module 4
│   └── strategies/
│       ├── __init__.py
│       ├── base_strategy.py         # Abstract contract
│       ├── fixed_strategy.py        # Fixed implementation
│       ├── fifo_strategy.py         # FIFO implementation
│       └── utils/
│           └── __init__.py          # Shared utilities
├── tests/
│   ├── test_data_generation.py
│   └── test_data_distribution.py
└── outputs/
    ├── wait_distribution.png
    ├── utilization.png
    ├── overrun_histogram.png
    └── paired_difference.png
```

---

## Extension Points

### Adding a New Strategy

1. Create new file: `src/strategies/my_strategy.py`
2. Inherit from `SchedulingStrategy`
3. Implement `process_shift()` and `name` property
4. Use utilities from `src/strategies/utils/`
5. Register in `monte_carlo_batcher._resolve_strategies()`

### Modifying Simulation Parameters

1. Update `SimulationConfig` defaults in `src/config.py`
2. Pass overrides via `config.with_overrides(...)`
3. No strategy code changes required

### Adding New Metrics

1. Extend `ShiftStatistics` dataclass in `src/models.py`
2. Update calculation in strategy or utils
3. Visualizer automatically picks up new fields via Schema S

---

## Testing Strategy

### Unit Tests
- Test individual utility functions
- Verify scenario generation determinism
- Validate statistics calculations

### Integration Tests
- End-to-end pipeline execution
- Verify paired-difference structure
- Check visualization output generation

### Statistical Tests
- Verify confidence interval calculations
- Validate paired t-test implementation

---

## Performance Considerations

1. **Slots in dataclasses:** Reduces memory footprint for large Monte Carlo runs
2. **Immutable structures:** Enables potential parallelization
3. **Early termination:** Safety limits prevent infinite loops
4. **Efficient data structures:** Dicts for O(1) machine lookups

---

## Conclusion

This four-tier architecture provides a robust, extensible framework for evaluating scheduling heuristics in resource-constrained environments. The strict separation between scenario generation, strategy execution, and analysis ensures statistical validity while maintaining code clarity and modularity.

The bipartite resource constraint model captures realistic operational dynamics, and the polymorphic strategy interface enables easy comparison of new algorithms against established baselines.
