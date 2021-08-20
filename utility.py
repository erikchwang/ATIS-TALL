import argparse, bz2, conlleval, csv, datetime, functools, glob, itertools, json, logging, multiprocessing, nltk
import os, pickle, random, stanza, torch, transformers, warnings

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "source")
target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "target")
task_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "task")
transformers_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "transformers")
source_mask_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "source_mask")
source_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "source_train")
source_develop_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "source_develop")
target_mask_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "target_mask")
target_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "target_train")
target_develop_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "target_develop")
slot_vocabulary_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "slot_vocabulary")
intent_vocabulary_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intent_vocabulary")
task_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "task_train")
task_develop_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "task_develop")
task_test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "task_test")
pre_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "pre_train")
fine_tune_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "fine_tune")
base_line_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "base_line")
source_train_example_count = 1000000
source_develop_example_count = 10000
target_train_example_count = 1000000
target_develop_example_count = 10000
file_queue_stride_size = 32
per_device_batch_size = 32
per_device_worker_count = 2
token_array_minimum_size = 8
token_array_maximum_size = 32
token_array_dropout_rate = 0.1
token_array_shuffle_size = 4
umt_model_learning_rate = 0.0001
nlu_model_learning_rate = 0.00005
adamw_optimizer_epsilon_value = 0.000001
adamw_optimizer_weight_decay = 0.01
learning_rate_decay_rate = 0.5
early_stopping_round_limit = 3
weight_decay_skip_terms = ["bias", "norm"]


def load_file(file_path, file_type):
    if file_type == "pickle":
        with open(file_path, "rb") as stream:
            return pickle.load(stream)

    elif file_type == "json":
        with open(file_path, "rt") as stream:
            return json.load(stream)

    elif file_type == "text":
        with open(file_path, "rt") as stream:
            return stream.read().splitlines()

    elif file_type == "bz2":
        with bz2.open(file_path, "rt") as stream:
            return stream.read().splitlines()

    else:
        raise Exception("invalid file type: {}".format(file_type))


def dump_file(file_items, file_path, file_type):
    if file_type == "pickle":
        with open(file_path, "wb") as stream:
            pickle.dump(file_items, stream)

    elif file_type == "json":
        with open(file_path, "wt") as stream:
            json.dump(file_items, stream)

    elif file_type == "text":
        with open(file_path, "wt") as stream:
            stream.write("\n".join(file_items))

    elif file_type == "bz2":
        with bz2.open(file_path, "wt") as stream:
            stream.write("\n".join(file_items))

    else:
        raise Exception("invalid file type: {}".format(file_type))


def convert_wikipedia(wikipedia_record, sentence_splitter, wordpiece_tokenizer):
    wikipedia_examples = []

    try:
        wikipedia_document = sentence_splitter(
            "\n\n".join(
                paragraph
                for paragraph in json.loads(wikipedia_record)["text"].splitlines()
                if len(paragraph.strip()) != 0
            )
        )

    except:
        wikipedia_document = None

    if wikipedia_document is not None:
        for sentence in wikipedia_document.sentences:
            token_array = wordpiece_tokenizer.encode(sentence.text, add_special_tokens=False)

            if token_array_minimum_size <= len(token_array) <= token_array_maximum_size:
                wikipedia_example = {"token_array": token_array}
                wikipedia_examples.append(wikipedia_example)

    return wikipedia_examples


def convert_atis(atis_record, wordpiece_tokenizer, slot_vocabulary, intent_vocabulary):
    token_array = []
    index_array = []

    for index, word in enumerate(atis_record["utterance"].split()):
        for token in wordpiece_tokenizer.encode(word, add_special_tokens=False):
            token_array.append(token)
            index_array.append(index)

    if len(index_array) != 0:
        try:
            slot_label = list(slot_vocabulary.index(slot) for slot in atis_record["slot_labels"].split())

        except:
            slot_label = None

        if slot_label is not None and len(slot_label) == max(index_array) + 1:
            try:
                intent_label = intent_vocabulary.index(atis_record["intent"])

            except:
                intent_label = None

            if intent_label is not None:
                atis_example = {
                    "token_array": token_array,
                    "index_array": index_array,
                    "slot_label": slot_label,
                    "intent_label": intent_label
                }

                return atis_example


class WikipediaBatch:
    def __init__(self, token_arrays, token_counts, token_languages):
        self.token_arrays = token_arrays
        self.token_counts = token_counts
        self.token_languages = token_languages

    @classmethod
    def load_wikipedia(cls, wikipedia_examples, wikipedia_language):
        token_counts = list(len(example["token_array"]) for example in wikipedia_examples)
        maximum_count = max(token_counts)

        token_arrays = torch.stack(
            list(
                torch.cat(
                    [
                        torch.tensor(example["token_array"], dtype=torch.long),
                        torch.zeros([maximum_count - count], dtype=torch.long)
                    ]
                )
                for example, count in zip(wikipedia_examples, token_counts)
            )
        )

        token_counts = torch.tensor(token_counts, dtype=torch.long)
        token_languages = torch.full([len(wikipedia_examples)], wikipedia_language, dtype=torch.long)

        return cls(token_arrays, token_counts, token_languages)

    @classmethod
    def merge_wikipedia(cls, wikipedia_batches):
        token_counts = torch.cat(list(batch.token_counts for batch in wikipedia_batches))
        maximum_count = torch.max(token_counts).item()

        token_arrays = torch.cat(
            list(
                torch.cat(
                    [
                        batch.token_arrays,
                        torch.zeros(
                            [batch.token_arrays.size()[0], maximum_count - batch.token_arrays.size()[1]],
                            dtype=torch.long,
                            device=batch.token_arrays.device
                        )
                    ],
                    1
                )
                for batch in wikipedia_batches
            )
        )

        token_languages = torch.cat(list(batch.token_languages for batch in wikipedia_batches))

        return cls(token_arrays, token_counts, token_languages)

    def add_eos(self, eos_token):
        self.output_labels = torch.scatter(
            torch.cat(
                [
                    self.token_arrays,
                    torch.zeros([self.token_arrays.size()[0], 1], dtype=torch.long, device=self.token_arrays.device)
                ],
                1
            ),
            1,
            torch.unsqueeze(self.token_counts, 1),
            torch.full([self.token_arrays.size()[0], 1], eos_token, dtype=torch.long, device=self.token_arrays.device)
        )

        return self

    def pin_memory(self):
        self.token_arrays = self.token_arrays.pin_memory()
        self.token_counts = self.token_counts.pin_memory()
        self.token_languages = self.token_languages.pin_memory()

        return self

    def to(self, *args, **kwargs):
        self.token_arrays = self.token_arrays.to(*args, **kwargs)
        self.token_counts = self.token_counts.to(*args, **kwargs)
        self.token_languages = self.token_languages.to(*args, **kwargs)

        return self


class ATISBatch:
    def __init__(self, token_arrays, token_counts, token_languages, index_arrays, slot_labels, intent_labels):
        self.token_arrays = token_arrays
        self.token_counts = token_counts
        self.token_languages = token_languages
        self.index_arrays = index_arrays
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels

    @classmethod
    def load_atis(cls, atis_examples, atis_language):
        token_counts = list(len(example["token_array"]) for example in atis_examples)
        maximum_count = max(token_counts)

        token_arrays = torch.stack(
            list(
                torch.cat(
                    [
                        torch.tensor(example["token_array"], dtype=torch.long),
                        torch.zeros([maximum_count - count], dtype=torch.long)
                    ]
                )
                for example, count in zip(atis_examples, token_counts)
            )
        )

        index_arrays = torch.stack(
            list(
                torch.cat(
                    [
                        torch.tensor(example["index_array"], dtype=torch.long),
                        torch.zeros([maximum_count - count], dtype=torch.long)
                    ]
                )
                for example, count in zip(atis_examples, token_counts)
            )
        )

        token_counts = torch.tensor(token_counts, dtype=torch.long)
        token_languages = torch.full([len(atis_examples)], atis_language, dtype=torch.long)

        slot_labels = torch.stack(
            list(
                torch.cat(
                    [
                        torch.tensor(example["slot_label"], dtype=torch.long),
                        torch.zeros([maximum_count - len(example["slot_label"])], dtype=torch.long)
                    ]
                )
                for example in atis_examples
            )
        )

        intent_labels = torch.tensor(list(example["intent_label"] for example in atis_examples), dtype=torch.long)

        return cls(token_arrays, token_counts, token_languages, index_arrays, slot_labels, intent_labels)

    def pin_memory(self):
        self.token_arrays = self.token_arrays.pin_memory()
        self.token_counts = self.token_counts.pin_memory()
        self.token_languages = self.token_languages.pin_memory()
        self.index_arrays = self.index_arrays.pin_memory()
        self.slot_labels = self.slot_labels.pin_memory()
        self.intent_labels = self.intent_labels.pin_memory()

        return self

    def to(self, *args, **kwargs):
        self.token_arrays = self.token_arrays.to(*args, **kwargs)
        self.token_counts = self.token_counts.to(*args, **kwargs)
        self.token_languages = self.token_languages.to(*args, **kwargs)
        self.index_arrays = self.index_arrays.to(*args, **kwargs)
        self.slot_labels = self.slot_labels.to(*args, **kwargs)
        self.intent_labels = self.intent_labels.to(*args, **kwargs)

        return self


class UMTEncoder(torch.nn.Module):
    def __init__(self, context_encoder):
        super().__init__()
        self.context_encoder = context_encoder

    def forward(self, input_arrays, input_counts):
        memory_arrays = self.context_encoder(
            input_arrays,
            torch.lt(
                torch.unsqueeze(torch.arange(input_arrays.size()[1], dtype=torch.long, device=input_arrays.device), 0),
                torch.unsqueeze(input_counts, 1)
            ).float()
        )[0]

        memory_counts = input_counts

        return memory_arrays, memory_counts


class UMTDecoder(torch.nn.Module):
    def __init__(self, language_embedder, context_decoder):
        super().__init__()
        self.language_embedder = language_embedder
        self.context_decoder = context_decoder

    def forward(self, prompt_arrays, prompt_counts, prompt_languages, memory_arrays, memory_counts, require_shift):
        summary_arrays = self.context_decoder(
            attention_mask=torch.le(
                torch.unsqueeze(
                    torch.arange(prompt_arrays.size()[1] + 1, dtype=torch.long, device=prompt_arrays.device),
                    0
                ),
                torch.unsqueeze(prompt_counts, 1)
            ).float(),
            inputs_embeds=torch.cat(
                [
                    torch.unsqueeze(self.language_embedder(prompt_languages), 1),
                    self.context_decoder.get_input_embeddings()(prompt_arrays)
                ],
                1
            ),
            encoder_hidden_states=memory_arrays,
            encoder_attention_mask=torch.lt(
                torch.unsqueeze(
                    torch.arange(memory_arrays.size()[1], dtype=torch.long, device=memory_arrays.device),
                    0
                ),
                torch.unsqueeze(memory_counts, 1)
            ).float()
        )[0]

        if require_shift:
            summary_arrays = summary_arrays[:, :-1, :]
            summary_counts = prompt_counts

        else:
            summary_counts = torch.add(prompt_counts, torch.tensor(1, dtype=torch.long, device=prompt_counts.device))

        return summary_arrays, summary_counts


class UMTModel(torch.nn.Module):
    def __init__(self, umt_encoder, umt_decoder, output_predictor):
        super().__init__()
        self.umt_encoder = umt_encoder
        self.umt_decoder = umt_decoder
        self.output_predictor = output_predictor

    def forward(self, input_arrays, input_counts, prompt_arrays, prompt_counts, prompt_languages):
        memory_arrays, memory_counts = self.umt_encoder(input_arrays, input_counts)

        summary_arrays, summary_counts = self.umt_decoder(
            prompt_arrays, prompt_counts, prompt_languages,
            memory_arrays, memory_counts, False
        )

        output_predictions = self.output_predictor(summary_arrays)
        output_counts = summary_counts

        return output_predictions, output_counts


class NLUPredictor(torch.nn.Module):
    def __init__(self, slot_predictor, intent_predictor):
        super().__init__()
        self.slot_predictor = slot_predictor
        self.intent_predictor = intent_predictor

    def forward(self, summary_arrays, summary_counts, index_arrays):
        aggregation_weight = torch.masked_fill(
            torch.ones(
                [summary_arrays.size()[0], summary_arrays.size()[1]],
                dtype=torch.float,
                device=summary_arrays.device
            ),
            torch.ge(
                torch.unsqueeze(
                    torch.arange(summary_arrays.size()[1], dtype=torch.long, device=summary_arrays.device),
                    0
                ),
                torch.unsqueeze(summary_counts, 1)
            ),
            torch.tensor(0.0, dtype=torch.float, device=summary_arrays.device)
        )

        slot_predictions = self.slot_predictor(
            torch.scatter_add(
                torch.zeros(summary_arrays.size(), dtype=torch.float, device=summary_arrays.device),
                1,
                torch.repeat_interleave(torch.unsqueeze(index_arrays, 2), summary_arrays.size()[2], 2),
                torch.mul(
                    summary_arrays,
                    torch.unsqueeze(
                        torch.div(
                            aggregation_weight,
                            torch.gather(
                                torch.scatter_add(
                                    torch.zeros(
                                        aggregation_weight.size(),
                                        dtype=torch.float,
                                        device=aggregation_weight.device
                                    ),
                                    1,
                                    index_arrays,
                                    aggregation_weight
                                ),
                                1,
                                index_arrays
                            )
                        ),
                        2
                    )
                )
            )
        )

        slot_counts = torch.add(
            torch.max(index_arrays, 1)[0],
            torch.tensor(1, dtype=torch.long, device=index_arrays.device)
        )

        intent_predictions = self.intent_predictor(
            torch.sum(
                torch.mul(
                    summary_arrays,
                    torch.unsqueeze(torch.div(aggregation_weight, torch.sum(aggregation_weight, 1, True)), 2)
                ),
                1
            )
        )

        return slot_predictions, slot_counts, intent_predictions


class NLUModel(torch.nn.Module):
    def __init__(self, umt_encoder, umt_decoder, nlu_predictor):
        super().__init__()
        self.umt_encoder = umt_encoder
        self.umt_decoder = umt_decoder
        self.nlu_predictor = nlu_predictor

    def forward(self, input_arrays, input_counts, prompt_arrays, prompt_counts, prompt_languages, index_arrays):
        memory_arrays, memory_counts = self.umt_encoder(input_arrays, input_counts)

        if self.umt_decoder is None:
            summary_arrays = memory_arrays
            summary_counts = memory_counts

        else:
            summary_arrays, summary_counts = self.umt_decoder(
                prompt_arrays, prompt_counts, prompt_languages,
                memory_arrays, memory_counts, True
            )

        slot_predictions, slot_counts, intent_predictions = self.nlu_predictor(
            summary_arrays, summary_counts,
            index_arrays
        )

        return slot_predictions, slot_counts, intent_predictions


class ParallelWrapper(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, attribute_name):
        try:
            return super().__getattr__(attribute_name)

        except:
            return getattr(self.module, attribute_name)


class NoiseMaker:
    def __init__(self, dropout_rate, shuffle_size):
        self.dropout_rate = dropout_rate
        self.shuffle_size = shuffle_size

    def __call__(self, wikipedia_batch):
        selection_mask = torch.logical_or(
            torch.ge(
                torch.unsqueeze(
                    torch.arange(
                        wikipedia_batch.token_arrays.size()[1],
                        dtype=torch.long,
                        device=wikipedia_batch.token_arrays.device
                    ),
                    0
                ),
                torch.unsqueeze(wikipedia_batch.token_counts, 1)
            ),
            torch.lt(
                torch.rand(
                    wikipedia_batch.token_arrays.size(),
                    dtype=torch.float,
                    device=wikipedia_batch.token_arrays.device
                ),
                torch.tensor(self.dropout_rate, dtype=torch.float, device=wikipedia_batch.token_arrays.device)
            )
        )

        token_arrays = torch.gather(
            torch.masked_fill(
                wikipedia_batch.token_arrays,
                selection_mask,
                torch.tensor(0, dtype=torch.long, device=selection_mask.device)
            ),
            1,
            torch.argsort(
                torch.masked_fill(
                    torch.add(
                        torch.unsqueeze(
                            torch.arange(
                                wikipedia_batch.token_arrays.size()[1],
                                dtype=torch.float,
                                device=wikipedia_batch.token_arrays.device
                            ),
                            0
                        ),
                        torch.randint(
                            self.shuffle_size,
                            wikipedia_batch.token_arrays.size(),
                            dtype=torch.float,
                            device=wikipedia_batch.token_arrays.device
                        )
                    ),
                    selection_mask,
                    torch.tensor(float("inf"), dtype=torch.float, device=selection_mask.device)
                )
            )
        )

        token_counts = torch.sum(torch.logical_not(selection_mask), 1, dtype=torch.long)
        token_arrays = token_arrays[:, :torch.max(token_counts).item()]
        token_languages = wikipedia_batch.token_languages

        return WikipediaBatch(token_arrays, token_counts, token_languages)


class MatchMaker:
    def __init__(self, umt_model, source_mask, target_mask, eos_token, minimum_size, maximum_size):
        self.umt_model = umt_model
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.eos_token = eos_token
        self.minimum_size = minimum_size
        self.maximum_size = maximum_size

    def __call__(self, wikipedia_batch):
        input_arrays = wikipedia_batch.token_arrays
        input_counts = wikipedia_batch.token_counts
        memory_arrays, memory_counts = self.umt_model.umt_encoder(input_arrays, input_counts)

        token_arrays = torch.empty(
            [wikipedia_batch.token_arrays.size()[0], 0],
            dtype=torch.long,
            device=wikipedia_batch.token_arrays.device
        )

        token_counts = torch.zeros(
            wikipedia_batch.token_counts.size(),
            dtype=torch.long,
            device=wikipedia_batch.token_counts.device
        )

        token_languages = torch.sub(
            torch.tensor(1, dtype=torch.long, device=wikipedia_batch.token_languages.device),
            wikipedia_batch.token_languages
        )

        vocabulary_mask = torch.index_select(
            torch.tensor([self.source_mask, self.target_mask], dtype=torch.bool, device=token_languages.device),
            0,
            token_languages
        )

        for index in range(self.maximum_size):
            prompt_arrays = token_arrays
            prompt_counts = token_counts
            prompt_languages = token_languages

            summary_arrays, summary_counts = self.umt_model.umt_decoder(
                prompt_arrays, prompt_counts, prompt_languages,
                memory_arrays, memory_counts, False
            )

            output_predictions = self.umt_model.output_predictor(summary_arrays)
            output_counts = summary_counts

            if index < self.minimum_size:
                prediction_mask = vocabulary_mask

            else:
                prediction_mask = torch.scatter(
                    vocabulary_mask,
                    1,
                    torch.full(
                        [output_predictions.size()[0], 1],
                        self.eos_token,
                        dtype=torch.long,
                        device=output_predictions.device
                    ),
                    torch.full(
                        [output_predictions.size()[0], 1],
                        False,
                        dtype=torch.bool,
                        device=output_predictions.device
                    )
                )

            token_generations = torch.argmax(
                torch.masked_fill(
                    output_predictions[:, -1, :],
                    prediction_mask,
                    torch.tensor(float("-inf"), dtype=torch.float, device=prediction_mask.device)
                ),
                1,
                True
            )

            if index < self.minimum_size:
                token_arrays = torch.cat([token_arrays, token_generations], 1)
                token_counts = output_counts

            else:
                generation_mask = torch.logical_or(
                    torch.ne(
                        torch.unsqueeze(token_counts, 1),
                        torch.tensor(index, dtype=torch.long, device=token_counts.device)
                    ),
                    torch.eq(
                        token_generations,
                        torch.tensor(self.eos_token, dtype=torch.long, device=token_generations.device)
                    )
                )

                if torch.all(generation_mask).item():
                    break

                else:
                    token_arrays = torch.cat(
                        [
                            token_arrays,
                            torch.masked_fill(
                                token_generations,
                                generation_mask,
                                torch.tensor(0, dtype=torch.long, device=generation_mask.device)
                            )
                        ],
                        1
                    )

                    token_counts = torch.add(
                        token_counts,
                        torch.squeeze(torch.logical_not(generation_mask).long(), 1)
                    )

        return WikipediaBatch(token_arrays, token_counts, token_languages)


def build_umt(umt_device):
    model_config = transformers.AutoConfig.from_pretrained(transformers_path)
    context_encoder = transformers.AutoModel.from_pretrained(transformers_path)
    context_encoder.get_input_embeddings().requires_grad_(False)
    umt_encoder = UMTEncoder(context_encoder)
    model_config.update({"is_decoder": True})
    model_config.update({"add_cross_attention": True})
    language_embedder = torch.nn.Embedding(2, model_config.hidden_size)
    context_decoder = transformers.AutoModel.from_config(model_config)
    context_decoder.get_input_embeddings().weight = context_encoder.get_input_embeddings().weight
    umt_decoder = UMTDecoder(language_embedder, context_decoder)
    output_predictor = torch.nn.Linear(model_config.hidden_size, model_config.vocab_size)
    output_predictor.weight = context_decoder.get_input_embeddings().weight
    umt_model = UMTModel(umt_encoder, umt_decoder, output_predictor)
    umt_model.to(umt_device)

    umt_optimizer = torch.optim.AdamW(
        [
            {
                "params": list(
                    parameter
                    for name, parameter in umt_model.named_parameters()
                    if all(term not in name.lower() for term in weight_decay_skip_terms)
                )
            },
            {
                "params": list(
                    parameter
                    for name, parameter in umt_model.named_parameters()
                    if any(term in name.lower() for term in weight_decay_skip_terms)
                ),
                "weight_decay": 0.0
            }
        ],
        umt_model_learning_rate,
        eps=adamw_optimizer_epsilon_value,
        weight_decay=adamw_optimizer_weight_decay
    )

    umt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(umt_optimizer, "max", learning_rate_decay_rate, 0)

    return umt_model, umt_optimizer, umt_scheduler


def build_nlu(nlu_device, require_decoder):
    slot_vocabulary = load_file(slot_vocabulary_path, "text")
    intent_vocabulary = load_file(intent_vocabulary_path, "text")
    model_config = transformers.AutoConfig.from_pretrained(transformers_path)
    context_encoder = transformers.AutoModel.from_pretrained(transformers_path)
    context_encoder.get_input_embeddings().requires_grad_(False)
    umt_encoder = UMTEncoder(context_encoder)

    if require_decoder:
        model_config.update({"is_decoder": True})
        model_config.update({"add_cross_attention": True})
        language_embedder = torch.nn.Embedding(2, model_config.hidden_size)
        context_decoder = transformers.AutoModel.from_config(model_config)
        context_decoder.get_input_embeddings().weight = context_encoder.get_input_embeddings().weight
        umt_decoder = UMTDecoder(language_embedder, context_decoder)

    else:
        umt_decoder = None

    slot_predictor = torch.nn.Linear(model_config.hidden_size, len(slot_vocabulary))
    intent_predictor = torch.nn.Linear(model_config.hidden_size, len(intent_vocabulary))
    nlu_predictor = NLUPredictor(slot_predictor, intent_predictor)
    nlu_model = NLUModel(umt_encoder, umt_decoder, nlu_predictor)
    nlu_model.to(nlu_device)

    nlu_optimizer = torch.optim.AdamW(
        [
            {
                "params": list(
                    parameter
                    for name, parameter in nlu_model.named_parameters()
                    if all(term not in name.lower() for term in weight_decay_skip_terms)
                )
            },
            {
                "params": list(
                    parameter
                    for name, parameter in nlu_model.named_parameters()
                    if any(term in name.lower() for term in weight_decay_skip_terms)
                ),
                "weight_decay": 0.0
            }
        ],
        nlu_model_learning_rate,
        eps=adamw_optimizer_epsilon_value,
        weight_decay=adamw_optimizer_weight_decay
    )

    nlu_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(nlu_optimizer, "max", learning_rate_decay_rate, 0)

    return nlu_model, nlu_optimizer, nlu_scheduler


def update_umt(umt_device, umt_model, umt_optimizer, source_data_loader, target_data_loader):
    source_mask = load_file(source_mask_path, "pickle")
    target_mask = load_file(target_mask_path, "pickle")
    wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)
    eos_token = wordpiece_tokenizer.encode("")[1]
    noise_maker = NoiseMaker(token_array_dropout_rate, token_array_shuffle_size)

    match_maker = MatchMaker(
        umt_model,
        source_mask, target_mask, eos_token,
        token_array_minimum_size, token_array_maximum_size
    )

    loss_criterion = torch.nn.CrossEntropyLoss()

    for source_data_batch, target_data_batch in zip(source_data_loader, target_data_loader):
        umt_model.eval()

        with torch.no_grad():
            source_data_batch.to(umt_device, non_blocking=True)
            target_data_batch.to(umt_device, non_blocking=True)
            source_noise_batch = noise_maker(source_data_batch)
            target_noise_batch = noise_maker(target_data_batch)
            source_match_batch = match_maker(source_data_batch)
            target_match_batch = match_maker(target_data_batch)

        umt_model.train()

        input_batch = WikipediaBatch.merge_wikipedia(
            [
                source_noise_batch,
                target_noise_batch,
                source_match_batch,
                target_match_batch
            ]
        )

        prompt_batch = WikipediaBatch.merge_wikipedia(
            [
                source_data_batch,
                target_data_batch,
                source_data_batch,
                target_data_batch
            ]
        )

        input_arrays = input_batch.token_arrays
        input_counts = input_batch.token_counts
        prompt_arrays = prompt_batch.token_arrays
        prompt_counts = prompt_batch.token_counts
        prompt_languages = prompt_batch.token_languages

        output_predictions, output_counts = umt_model(
            input_arrays, input_counts,
            prompt_arrays, prompt_counts, prompt_languages
        )

        prompt_batch.add_eos(eos_token)
        output_labels = prompt_batch.output_labels
        loss_value = torch.tensor(0.0, dtype=torch.float, device=umt_device)

        for predictions_chunk, counts_chunk, labels_chunk in zip(
                torch.split(output_predictions, output_predictions.size()[0] // 4),
                torch.split(output_counts, output_counts.size()[0] // 4),
                torch.split(output_labels, output_labels.size()[0] // 4)
        ):
            loss_value = torch.add(
                loss_value,
                loss_criterion(
                    torch.reshape(predictions_chunk, [-1, predictions_chunk.size()[2]]),
                    torch.reshape(
                        torch.masked_fill(
                            labels_chunk,
                            torch.ge(
                                torch.unsqueeze(
                                    torch.arange(labels_chunk.size()[1], dtype=torch.long, device=umt_device),
                                    0
                                ),
                                torch.unsqueeze(counts_chunk, 1)
                            ),
                            torch.tensor(loss_criterion.ignore_index, dtype=torch.long, device=umt_device)
                        ),
                        [-1]
                    )
                )
            )

        umt_optimizer.zero_grad()
        loss_value.backward()
        umt_optimizer.step()


def update_nlu(nlu_device, nlu_model, nlu_optimizer, task_data_loader):
    loss_criterion = torch.nn.CrossEntropyLoss()
    nlu_model.train()

    for task_data_batch in task_data_loader:
        with torch.no_grad():
            task_data_batch.to(nlu_device, non_blocking=True)

        input_arrays = task_data_batch.token_arrays
        input_counts = task_data_batch.token_counts
        prompt_arrays = task_data_batch.token_arrays
        prompt_counts = task_data_batch.token_counts
        prompt_languages = task_data_batch.token_languages
        index_arrays = task_data_batch.index_arrays

        slot_predictions, slot_counts, intent_predictions = nlu_model(
            input_arrays, input_counts,
            prompt_arrays, prompt_counts, prompt_languages,
            index_arrays
        )

        slot_labels = task_data_batch.slot_labels
        intent_labels = task_data_batch.intent_labels

        loss_value = torch.add(
            loss_criterion(
                torch.reshape(slot_predictions, [-1, slot_predictions.size()[2]]),
                torch.reshape(
                    torch.masked_fill(
                        slot_labels,
                        torch.ge(
                            torch.unsqueeze(
                                torch.arange(slot_labels.size()[1], dtype=torch.long, device=nlu_device),
                                0
                            ),
                            torch.unsqueeze(slot_counts, 1)
                        ),
                        torch.tensor(loss_criterion.ignore_index, dtype=torch.long, device=nlu_device)
                    ),
                    [-1]
                )
            ),
            loss_criterion(intent_predictions, intent_labels)
        )

        nlu_optimizer.zero_grad()
        loss_value.backward()
        nlu_optimizer.step()


def assess_umt(umt_device, umt_model, source_data_loader, target_data_loader):
    source_mask = load_file(source_mask_path, "pickle")
    target_mask = load_file(target_mask_path, "pickle")
    wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)
    eos_token = wordpiece_tokenizer.encode("")[1]

    match_maker = MatchMaker(
        umt_model,
        source_mask, target_mask, eos_token,
        token_array_minimum_size, token_array_maximum_size
    )

    umt_model.eval()
    reference_inventory = []
    hypothesis_inventory = []

    for source_data_batch, target_data_batch in zip(source_data_loader, target_data_loader):
        with torch.no_grad():
            source_data_batch.to(umt_device, non_blocking=True)
            target_data_batch.to(umt_device, non_blocking=True)
            source_rematch_batch = match_maker(match_maker(source_data_batch))
            target_rematch_batch = match_maker(match_maker(target_data_batch))

            for token_array, token_count in zip(
                    source_data_batch.token_arrays.tolist() + target_data_batch.token_arrays.tolist(),
                    source_data_batch.token_counts.tolist() + target_data_batch.token_counts.tolist()
            ):
                reference_inventory.append([token_array[:token_count]])

            for token_array, token_count in zip(
                    source_rematch_batch.token_arrays.tolist() + target_rematch_batch.token_arrays.tolist(),
                    source_rematch_batch.token_counts.tolist() + target_rematch_batch.token_counts.tolist()
            ):
                hypothesis_inventory.append(token_array[:token_count])

    bleu_score = nltk.translate.bleu_score.corpus_bleu(reference_inventory, hypothesis_inventory)
    print("BLEU Score: {}".format(bleu_score))

    return bleu_score


def assess_nlu(nlu_device, nlu_model, task_data_loader):
    slot_vocabulary = load_file(slot_vocabulary_path, "text")
    nlu_model.eval()
    slot_inventory = []
    intent_inventory = []
    semantic_inventory = []

    for task_data_batch in task_data_loader:
        with torch.no_grad():
            task_data_batch.to(nlu_device, non_blocking=True)
            input_arrays = task_data_batch.token_arrays
            input_counts = task_data_batch.token_counts
            prompt_arrays = task_data_batch.token_arrays
            prompt_counts = task_data_batch.token_counts
            prompt_languages = task_data_batch.token_languages
            index_arrays = task_data_batch.index_arrays

            slot_predictions, slot_counts, intent_predictions = nlu_model(
                input_arrays, input_counts,
                prompt_arrays, prompt_counts, prompt_languages,
                index_arrays
            )

            slot_labels = task_data_batch.slot_labels
            intent_labels = task_data_batch.intent_labels
            slot_classes = torch.argmax(slot_predictions, 2)
            intent_classes = torch.argmax(intent_predictions, 1)

            for slot_count, slot_label, intent_label, slot_class, intent_class in zip(
                    slot_counts.tolist(),
                    slot_labels.tolist(),
                    intent_labels.tolist(),
                    slot_classes.tolist(),
                    intent_classes.tolist()
            ):
                for label_item, class_item in zip(slot_label[:slot_count], slot_class[:slot_count]):
                    slot_inventory.append(" ".join(["_", slot_vocabulary[label_item], slot_vocabulary[class_item]]))

                slot_inventory.append("")

                if intent_label == intent_class:
                    intent_inventory.append(1)

                    if slot_label[:slot_count] == slot_class[:slot_count]:
                        semantic_inventory.append(1)

                    else:
                        semantic_inventory.append(0)

                else:
                    intent_inventory.append(0)
                    semantic_inventory.append(0)

    slot_f1 = conlleval.evaluate(slot_inventory)["overall"]["chunks"]["evals"]["f1"]
    intent_accuracy = sum(intent_inventory) / len(intent_inventory)
    semantic_accuracy = sum(semantic_inventory) / len(semantic_inventory)
    print("Slot F1: {}, Intent Accuracy: {}, Semantic Accuracy: {}".format(slot_f1, intent_accuracy, semantic_accuracy))

    return semantic_accuracy
