# Final figures

All figures come from EPANET/WNTR synthetic scenarios and the saved model results. Regenerate with `python scripts/generate_final_reports.py`.

- `01_model_comparison_test.png`: Test error per junction and risk F1 for all five forecasters.
- `02_training_curves.png`: Validation MAE per epoch for the three neural networks.
- `03_error_by_scenario_family.png`: Test MAE per scenario family for every model.
- `04_recommendation_outcomes.png`: Outcome of each EPANET simulated validation, by engine problem type.
- `05_network_topology.png`: The EPANET network: R1, PU1, J1-J4, P1-P4, V1.
- `06_gru_prediction_vs_actual.png`: GRU forecast against the EPANET value (test scenarios).
- `07_gru_risk_confusion_matrix.png`: Risk class of the GRU forecast against the EPANET risk class.
- `08_before_after_pressure.png`: EPANET simulated validation: pressures before and after.
- `09_before_after_violations.png`: EPANET simulated validation: violation counts before and after.

This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.
