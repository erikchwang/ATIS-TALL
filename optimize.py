from utility import *

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("step_index", type=int)
torch.distributed.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if argument_parser.parse_args().step_index == 0:
    source_train = load_file(source_train_path, "pickle")
    target_train = load_file(target_train_path, "pickle")
    source_develop = load_file(source_develop_path, "pickle")
    target_develop = load_file(target_develop_path, "pickle")
    umt_device = torch.device("cuda")
    umt_model, umt_optimizer, umt_scheduler = build_umt(umt_device)

    source_train_loader = torch.utils.data.DataLoader(
        source_train,
        per_device_batch_size // 4,
        sampler=torch.utils.data.distributed.DistributedSampler(source_train),
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(WikipediaBatch.load_wikipedia, wikipedia_language=0),
        pin_memory=True
    )

    target_train_loader = torch.utils.data.DataLoader(
        target_train,
        per_device_batch_size // 4,
        sampler=torch.utils.data.distributed.DistributedSampler(target_train),
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(WikipediaBatch.load_wikipedia, wikipedia_language=1),
        pin_memory=True
    )

    source_develop_loader = torch.utils.data.DataLoader(
        source_develop,
        per_device_batch_size,
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(WikipediaBatch.load_wikipedia, wikipedia_language=0),
        pin_memory=True
    )

    target_develop_loader = torch.utils.data.DataLoader(
        target_develop,
        per_device_batch_size,
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(WikipediaBatch.load_wikipedia, wikipedia_language=1),
        pin_memory=True
    )

    while True:
        torch.distributed.barrier()
        begin_time = datetime.datetime.now()
        pre_train = torch.load(pre_train_path, umt_device)

        if pre_train["round"] > early_stopping_round_limit:
            break

        umt_model.load_state_dict(pre_train["model"])
        umt_optimizer.load_state_dict(pre_train["optimizer"])
        umt_scheduler.load_state_dict(pre_train["scheduler"])
        source_train_loader.sampler.set_epoch(umt_scheduler.last_epoch)
        target_train_loader.sampler.set_epoch(umt_scheduler.last_epoch)
        umt_model = ParallelWrapper(umt_model, [umt_device], umt_device, find_unused_parameters=True)
        update_umt(umt_device, umt_model, umt_optimizer, source_train_loader, target_train_loader)
        umt_model = umt_model.module

        if torch.distributed.get_rank() == 0:
            bleu_score = assess_umt(umt_device, umt_model, source_develop_loader, target_develop_loader)

            if umt_scheduler.is_better(bleu_score, umt_scheduler.best):
                umt_result = True
                pre_train["model"] = umt_model.state_dict()

            else:
                umt_result = False
                pre_train["round"] += 1
                umt_optimizer.load_state_dict(pre_train["optimizer"])

            umt_scheduler.step(bleu_score)
            pre_train["optimizer"] = umt_optimizer.state_dict()
            pre_train["scheduler"] = umt_scheduler.state_dict()
            torch.save(pre_train, pre_train_path)

            print(
                "optimize {}: cost {} seconds in epoch {} and {}".format(
                    pre_train_path,
                    int((datetime.datetime.now() - begin_time).total_seconds()),
                    umt_scheduler.last_epoch,
                    "accept the result" if umt_result else "reject the result"
                )
            )

elif argument_parser.parse_args().step_index == 1:
    task_train = load_file(task_train_path, "pickle")
    task_develop = load_file(task_develop_path, "pickle")
    nlu_device = torch.device("cuda")
    nlu_model, nlu_optimizer, nlu_scheduler = build_nlu(nlu_device, True)

    task_train_loader = torch.utils.data.DataLoader(
        task_train,
        per_device_batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(task_train),
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(ATISBatch.load_atis, atis_language=0),
        pin_memory=True
    )

    task_develop_loader = torch.utils.data.DataLoader(
        task_develop,
        per_device_batch_size,
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(ATISBatch.load_atis, atis_language=1),
        pin_memory=True
    )

    while True:
        torch.distributed.barrier()
        begin_time = datetime.datetime.now()
        fine_tune = torch.load(fine_tune_path, nlu_device)

        if fine_tune["round"] > early_stopping_round_limit:
            break

        nlu_model.load_state_dict(fine_tune["model"])
        nlu_optimizer.load_state_dict(fine_tune["optimizer"])
        nlu_scheduler.load_state_dict(fine_tune["scheduler"])
        task_train_loader.sampler.set_epoch(nlu_scheduler.last_epoch)
        nlu_model = ParallelWrapper(nlu_model, [nlu_device], nlu_device, find_unused_parameters=True)
        update_nlu(nlu_device, nlu_model, nlu_optimizer, task_train_loader)
        nlu_model = nlu_model.module

        if torch.distributed.get_rank() == 0:
            semantic_accuracy = assess_nlu(nlu_device, nlu_model, task_develop_loader)

            if nlu_scheduler.is_better(semantic_accuracy, nlu_scheduler.best):
                nlu_result = True
                fine_tune["model"] = nlu_model.state_dict()

            else:
                nlu_result = False
                fine_tune["round"] += 1
                nlu_optimizer.load_state_dict(fine_tune["optimizer"])

            nlu_scheduler.step(semantic_accuracy)
            fine_tune["optimizer"] = nlu_optimizer.state_dict()
            fine_tune["scheduler"] = nlu_scheduler.state_dict()
            torch.save(fine_tune, fine_tune_path)

            print(
                "optimize {}: cost {} seconds in epoch {} and {}".format(
                    fine_tune_path,
                    int((datetime.datetime.now() - begin_time).total_seconds()),
                    nlu_scheduler.last_epoch,
                    "accept the result" if nlu_result else "reject the result"
                )
            )

elif argument_parser.parse_args().step_index == 2:
    task_train = load_file(task_train_path, "pickle")
    task_develop = load_file(task_develop_path, "pickle")
    nlu_device = torch.device("cuda")
    nlu_model, nlu_optimizer, nlu_scheduler = build_nlu(nlu_device, False)

    task_train_loader = torch.utils.data.DataLoader(
        task_train,
        per_device_batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(task_train),
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(ATISBatch.load_atis, atis_language=0),
        pin_memory=True
    )

    task_develop_loader = torch.utils.data.DataLoader(
        task_develop,
        per_device_batch_size,
        num_workers=per_device_worker_count,
        collate_fn=functools.partial(ATISBatch.load_atis, atis_language=1),
        pin_memory=True
    )

    while True:
        torch.distributed.barrier()
        begin_time = datetime.datetime.now()
        base_line = torch.load(base_line_path, nlu_device)

        if base_line["round"] > early_stopping_round_limit:
            break

        nlu_model.load_state_dict(base_line["model"])
        nlu_optimizer.load_state_dict(base_line["optimizer"])
        nlu_scheduler.load_state_dict(base_line["scheduler"])
        task_train_loader.sampler.set_epoch(nlu_scheduler.last_epoch)
        nlu_model = ParallelWrapper(nlu_model, [nlu_device], nlu_device, find_unused_parameters=True)
        update_nlu(nlu_device, nlu_model, nlu_optimizer, task_train_loader)
        nlu_model = nlu_model.module

        if torch.distributed.get_rank() == 0:
            semantic_accuracy = assess_nlu(nlu_device, nlu_model, task_develop_loader)

            if nlu_scheduler.is_better(semantic_accuracy, nlu_scheduler.best):
                nlu_result = True
                base_line["model"] = nlu_model.state_dict()

            else:
                nlu_result = False
                base_line["round"] += 1
                nlu_optimizer.load_state_dict(base_line["optimizer"])

            nlu_scheduler.step(semantic_accuracy)
            base_line["optimizer"] = nlu_optimizer.state_dict()
            base_line["scheduler"] = nlu_scheduler.state_dict()
            torch.save(base_line, base_line_path)

            print(
                "optimize {}: cost {} seconds in epoch {} and {}".format(
                    base_line_path,
                    int((datetime.datetime.now() - begin_time).total_seconds()),
                    nlu_scheduler.last_epoch,
                    "accept the result" if nlu_result else "reject the result"
                )
            )

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))
