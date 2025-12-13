def simple_scheduler(queue, run_inference, max_batch=32, latency_budget=0.02):
    batch = []
    while True:
        try:
            req = queue.get(timeout=latency_budget)
        except Exception:
            req = None
        if req is not None:
            batch.append(req)
        timeout = (len(batch) > 0 and (len(batch) >= max_batch))
        if timeout or (req is None and len(batch) > 0):
            run_inference(batch)
            batch = []
