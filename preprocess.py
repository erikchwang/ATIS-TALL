from utility import *

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("step_index", type=int)

if argument_parser.parse_args().step_index == 0:
    with multiprocessing.Pool() as pool:
        sentence_splitter = stanza.Pipeline(
            os.path.basename(os.path.dirname(glob.glob("{}/stanza/*/".format(source_path))[0])),
            "{}/stanza".format(source_path),
            processors="tokenize",
            use_gpu=False
        )

        wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)
        file_paths = glob.glob("{}/corpus/*/*".format(source_path))
        random.shuffle(file_paths)
        wikipedia_examples = []

        while len(wikipedia_examples) < source_train_example_count + source_develop_example_count:
            wikipedia_examples.extend(
                itertools.chain.from_iterable(
                    pool.map(
                        functools.partial(
                            convert_wikipedia,
                            sentence_splitter=sentence_splitter,
                            wordpiece_tokenizer=wordpiece_tokenizer
                        ),
                        itertools.chain.from_iterable(
                            load_file(file_paths.pop(), "bz2")
                            for _ in range(file_queue_stride_size)
                        )
                    )
                )
            )

        token_vocabulary = set(itertools.chain.from_iterable(example["token_array"] for example in wikipedia_examples))
        source_mask = list(index not in token_vocabulary for index in range(wordpiece_tokenizer.vocab_size))
        source_train = wikipedia_examples[:source_train_example_count]
        source_develop = wikipedia_examples[-source_develop_example_count:]
        dump_file(source_mask, source_mask_path, "pickle")
        dump_file(source_train, source_train_path, "pickle")
        dump_file(source_develop, source_develop_path, "pickle")

elif argument_parser.parse_args().step_index == 1:
    with multiprocessing.Pool() as pool:
        sentence_splitter = stanza.Pipeline(
            os.path.basename(os.path.dirname(glob.glob("{}/stanza/*/".format(target_path))[0])),
            "{}/stanza".format(target_path),
            processors="tokenize",
            use_gpu=False
        )

        wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)
        file_paths = glob.glob("{}/corpus/*/*".format(target_path))
        random.shuffle(file_paths)
        wikipedia_examples = []

        while len(wikipedia_examples) < target_train_example_count + target_develop_example_count:
            wikipedia_examples.extend(
                itertools.chain.from_iterable(
                    pool.map(
                        functools.partial(
                            convert_wikipedia,
                            sentence_splitter=sentence_splitter,
                            wordpiece_tokenizer=wordpiece_tokenizer
                        ),
                        itertools.chain.from_iterable(
                            load_file(file_paths.pop(), "bz2")
                            for _ in range(file_queue_stride_size)
                        )
                    )
                )
            )

        token_vocabulary = set(itertools.chain.from_iterable(example["token_array"] for example in wikipedia_examples))
        target_mask = list(index not in token_vocabulary for index in range(wordpiece_tokenizer.vocab_size))
        target_train = wikipedia_examples[:target_train_example_count]
        target_develop = wikipedia_examples[-target_develop_example_count:]
        dump_file(target_mask, target_mask_path, "pickle")
        dump_file(target_train, target_train_path, "pickle")
        dump_file(target_develop, target_develop_path, "pickle")

elif argument_parser.parse_args().step_index == 2:
    slot_count = {}
    intent_count = {}

    for path in glob.glob("{}/*/*".format(task_path)):
        for record in csv.DictReader(load_file(path, "text"), dialect=csv.excel_tab):
            for slot in record["slot_labels"].split():
                if slot in slot_count:
                    slot_count[slot] += 1

                else:
                    slot_count[slot] = 1

            if record["intent"] in intent_count:
                intent_count[record["intent"]] += 1

            else:
                intent_count[record["intent"]] = 1

    slot_vocabulary = sorted(slot_count, key=slot_count.get, reverse=True)
    intent_vocabulary = sorted(intent_count, key=intent_count.get, reverse=True)
    dump_file(slot_vocabulary, slot_vocabulary_path, "text")
    dump_file(intent_vocabulary, intent_vocabulary_path, "text")

elif argument_parser.parse_args().step_index == 3:
    with multiprocessing.Pool() as pool:
        slot_vocabulary = load_file(slot_vocabulary_path, "text")
        intent_vocabulary = load_file(intent_vocabulary_path, "text")
        wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)

        for subset_name, data_path in [
            ["train", task_train_path],
            ["develop", task_develop_path],
            ["test", task_test_path]
        ]:
            atis_examples = list(
                example
                for example in pool.map(
                    functools.partial(
                        convert_atis,
                        wordpiece_tokenizer=wordpiece_tokenizer,
                        slot_vocabulary=slot_vocabulary,
                        intent_vocabulary=intent_vocabulary
                    ),
                    itertools.chain.from_iterable(
                        csv.DictReader(load_file(path, "text"), dialect=csv.excel_tab)
                        for path in glob.glob("{}/{}/*".format(task_path, subset_name))
                    )
                )
                if example is not None
            )

            dump_file(atis_examples, data_path, "pickle")

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))

print(
    "preprocess step {}: cost {} seconds".format(
        argument_parser.parse_args().step_index,
        int((datetime.datetime.now() - begin_time).total_seconds())
    )
)
