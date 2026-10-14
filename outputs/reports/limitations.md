# Limitations

> This is a simulation-based decision-support prototype, not autonomous real-world utility control.

1. **Synthetic data only.** All data in this project is synthetic: it comes only from EPANET simulations run through WNTR. No real utility data is used. Real networks have noise, missing data, sensor drift, unknown demands and unmodelled equipment.
2. **Tiny network.** Four junctions, four pipes, one pump, one PRV, no tank. Behaviour, forecast difficulty and recommendation options are far simpler than a real district.
3. **Idealised scenarios.** Faults (leaks, blockage, valve misconfiguration, pump wear, high source head) are drawn from fixed families and ranges. Real faults are messier and can overlap.
4. **Optimistic validation.** The EPANET simulated validation re-runs the same simulator that produced the training data, so agreement is expected. This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.
5. **Lab-prototype thresholds.** Pressure 20–60 m, J4 minimum 30 m and spread 20 m are configurable teaching values, not utility standards.
6. **Short horizon.** Forecasts are one step (15 minutes) ahead from a 3-hour window; no multi-hour planning, no demand forecasting.
7. **Simple models win on this task.** Persistence is already strong and the random forest with engineered features beats the neural networks on test error; the networks' advantage on richer, noisier data is not shown here. Each network was trained with one main seed (see the seed check for the spread).
8. **Rule-based risk and recommendations.** Transparent by design, but the first-order what-if estimate can be wrong (for example for a blocked pipe); the EPANET re-run is what is trusted.
9. **Leak handling is minimal.** Leak-like patterns are flagged by a mass-balance check and always lead to inspection; leak localisation is only a suspicion.
10. **No hardware.** No SCADA, no physical sensors, no actuator interface, no latency, safety interlocks or operator workflow. Recommendations are suggestions in a simulation.
11. **Single critical node.** Only J4 has the stricter rule; real systems have many priority customers and conflicting constraints.
12. **Reservoir head choice.** The base reservoir head is 62 m (the Phase 1 value of 110 m left every consumer far above the pressure limit); 110 m is reached only in high-source-pressure scenarios.
