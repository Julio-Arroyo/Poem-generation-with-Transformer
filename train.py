from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, inputs: BatchEncoding, labels: BatchEncoding):
        """
        Inputs: encoded sequences of words.
        Labels: the next word in the sequence (also encoded)

        Usage: you can access them like lists, i.e. self.inputs[i]
        this will give you a tokenizers.Encoding object; among its
        useful attributes are ids (words encoded as numbers) and
        attention_mask (to feed into the model). Examples:
        >> self.inputs[i].ids
        >> self.inputs[i].attention_mask
        """
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.inputs[idx].ids)
        input_masks = torch.LongTensor(self.inputs[idx].attention_mask)
        label_ids = torch.LongTensor(self.labels[idx].ids)
        # print(f'INDIV. SHAPES: {input_ids.shape}, {input_masks.shape}, {label_ids.shape}')
        data = {'inputs_ids': input_ids,
                'inputs_mask': input_masks,
                'labels_ids': label_ids}
        return data

    def __len__(self):
        return len(self.labels)


def parse_corpus_text(text, seq_len, skip_size=1):
    """
    Takes as input a corpus of text as a single string.
    
    Parameters:
        - text: str
        - seq_len: (int), the number of characters you want per sequence in 
                        your dataset
        - skip_size: (int) the size of the jump between sequences.
    
    Returns:
        - X: list of sentences, each of length seq_len
        - Y: next word after each sentence
        - vocabulary: set of unique words
    """
    assert seq_len > 0, 'Training sequences must be of length at least 1'

    text = text.replace('\n', ' ')

    X = []
    Y = []
    vocabulary = set()

    words = text.split()

    i = 0
    while i < len(words) - 2*seq_len - 1:
        sentence = ' '.join(words[i:i+seq_len])
        next_word = ' '.join(words[i+seq_len: i+2*seq_len])
        vocabulary.update(words[i:i+seq_len])
        vocabulary.update(next_word)
        # vocabulary.add(next_word)
        X.append(sentence)
        if len(next_word) == 0:
            print(f'FOUND WORD OF LENGTH ZERO')
        Y.append(next_word)
        i += skip_size

    return X, Y, vocabulary


def encode_inputs_labels(X, Y, tokenizer):
    assert len(X) == len(Y), 'Lengths of sentences and targets are different'

    X_enc = []
    Y_enc = []
    for i in range(len(X)):
        sentence = X[i]
        next_word = Y[i]
        enc_input = tokenizer.encode(sentence)
        enc_target = tokenizer.encode(next_word)
        X_enc.append(enc_input)
        Y_enc.append(enc_target)
    X_enc = torch.tensor(X_enc)
    Y_enc = torch.tensor(Y_enc)
    return X_enc, Y_enc


def generate_poem(model, seed_str, num_stanzas=8, num_poems=5):
    for i in range(num_poems):
        print(f'POEM #{i}')
        poem = seed_str
        curr_seed = seed_str
        for j in range(num_stanzas):
            tokens = tokenizer.encode(curr_seed, return_tensors="pt").to(device)
            prediction = model.generate(tokens, min_length=40, max_length=48, do_sample=True, repetition_penalty=1.2)
            poem += f'({j}): \t{tokenizer.decode(prediction[0])[len(curr_seed):]}\n'
            curr_seed = tokenizer.decode(prediction[0])[len(curr_seed):]
        print(poem)


if __name__ == '__main__':
    # Make dataset: raw text -> sentences -> encoded inputs & targets
    raw_text = open('data/shakespeare.txt').read()
    seq_len = 20
    X, Y, vocabulary = parse_corpus_text(raw_text, seq_len)
    # config = GPT2Config(vocab_size=len(vocabulary))
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    (X_enc, Y_enc) = (tokenizer(X, truncation=True, padding=True), tokenizer(Y, truncation=True, padding=True))

    dataset = TextDataset(X_enc, Y_enc)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    model.to(device)

    # Hyper-parameters
    bs = 512
    lr = 0.0005
    num_epochs = 2

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=bs,
                                               shuffle=False)
    # print('POEM BEFORE TRAINING')
    # generate_poem(model, "Shall I compare thee to a summer day", num_poems=1)

    for curr_epoch in range(num_epochs):
        print(f'EPOCH #{curr_epoch}')
        tokens = tokenizer.encode("His tender heir might bear his memory:",
                                  return_tensors="pt").to(device)
        prediction = model.generate(tokens,
                                    min_length=20,
                                    max_length=40,
                                    do_sample=True,
                                    repetition_penalty=1.2,
                                    temperature=1.0)
        print(tokenizer.decode(prediction[0]))
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # from the tokenizers.Encoding objects, get what is needed
            sentences_ids = batch['inputs_ids']
            sentences_att_mask = batch['inputs_mask']
            target_ids = batch['labels_ids']

            # print(f'SHAPES: {sentences_ids.shape, sentences_att_mask.shape, target_ids.shape}')
            outputs = model(sentences_ids,
                            attention_mask=sentences_att_mask,
                            labels=target_ids)

            loss = outputs[0]
            loss.backward()
            optimizer.step()

    print()
    print('POEMS AFTER TRAINING')
    generate_poem(model, "Shall I compare thee to a summer day")
