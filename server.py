import flwr as fl
import numpy as np

# Define strategy with metrics aggregation
def weighted_average(metrics):
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {'accuracy': sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
)

# Start server
fl.server.start_server(
    server_address='localhost:8080',
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)