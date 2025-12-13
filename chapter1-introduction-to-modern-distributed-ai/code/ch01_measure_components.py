def measure_components(measure_data_loading_time, measure_compute_time, measure_communication_time):
    data_time = measure_data_loading_time()
    compute_time = measure_compute_time()
    comm_time = measure_communication_time()

    total_time = data_time + compute_time + comm_time
    print(f"Data loading: {data_time/total_time*100:.1f}%")
    print(f"Computation: {compute_time/total_time*100:.1f}%")
    print(f"Communication: {comm_time/total_time*100:.1f}%")
