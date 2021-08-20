from utility import *

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("step_index", type=int)

if argument_parser.parse_args().step_index == 0:
    task_test = load_file(task_test_path, "pickle")
    nlu_device = torch.device("cuda")
    nlu_model = build_nlu(nlu_device, True)[0]
    fine_tune = torch.load(fine_tune_path, nlu_device)
    nlu_model.load_state_dict(fine_tune["model"])

    task_test_loader = torch.utils.data.DataLoader(
        task_test,
        per_device_batch_size,
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(ATISBatch.load_atis, atis_language=1),
        pin_memory=True
    )

    print("execute {}: running the model".format(fine_tune_path))
    assess_nlu(nlu_device, nlu_model, task_test_loader)

elif argument_parser.parse_args().step_index == 1:
    task_test = load_file(task_test_path, "pickle")
    nlu_device = torch.device("cuda")
    nlu_model = build_nlu(nlu_device, False)[0]
    base_line = torch.load(base_line_path, nlu_device)
    nlu_model.load_state_dict(base_line["model"])

    task_test_loader = torch.utils.data.DataLoader(
        task_test,
        per_device_batch_size,
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(ATISBatch.load_atis, atis_language=1),
        pin_memory=True
    )

    print("execute {}: running the model".format(base_line_path))
    assess_nlu(nlu_device, nlu_model, task_test_loader)

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))
