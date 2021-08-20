from utility import *

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("step_index", type=int)

if argument_parser.parse_args().step_index == 0:
    umt_device = torch.device("cuda")
    umt_model, umt_optimizer, umt_scheduler = build_umt(umt_device)

    pre_train = {
        "round": 0,
        "model": umt_model.state_dict(),
        "optimizer": umt_optimizer.state_dict(),
        "scheduler": umt_scheduler.state_dict()
    }

    torch.save(pre_train, pre_train_path)

elif argument_parser.parse_args().step_index == 1:
    umt_device = torch.device("cuda")
    umt_model = build_umt(umt_device)[0]
    pre_train = torch.load(pre_train_path, umt_device)
    umt_model.load_state_dict(pre_train["model"])
    nlu_device = torch.device("cuda")
    nlu_model, nlu_optimizer, nlu_scheduler = build_nlu(nlu_device, True)
    nlu_model.umt_encoder.load_state_dict(umt_model.umt_encoder.state_dict())
    nlu_model.umt_decoder.load_state_dict(umt_model.umt_decoder.state_dict())

    fine_tune = {
        "round": 0,
        "model": nlu_model.state_dict(),
        "optimizer": nlu_optimizer.state_dict(),
        "scheduler": nlu_scheduler.state_dict()
    }

    torch.save(fine_tune, fine_tune_path)

elif argument_parser.parse_args().step_index == 2:
    nlu_device = torch.device("cuda")
    nlu_model, nlu_optimizer, nlu_scheduler = build_nlu(nlu_device, False)

    base_line = {
        "round": 0,
        "model": nlu_model.state_dict(),
        "optimizer": nlu_optimizer.state_dict(),
        "scheduler": nlu_scheduler.state_dict()
    }

    torch.save(base_line, base_line_path)

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))

print(
    "construct step {}: cost {} seconds".format(
        argument_parser.parse_args().step_index,
        int((datetime.datetime.now() - begin_time).total_seconds())
    )
)
