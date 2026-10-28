# Experiment settings

`paper.json` records the shared retrieval window, edit budget, PGD schedule,
first-order smoothness weight, and the Gradient-LSE temperature used in the 100-query
primary comparison. Cohort data, model assets, and local filesystem paths are supplied
outside the repository. It is a declarative record of the paper settings; callers pass
the corresponding values through `PGDConfig` and the runtime hooks.
