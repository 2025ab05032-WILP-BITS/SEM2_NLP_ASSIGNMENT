# =============================================================================
# Feed-Forward Neural Language Model — AG News Dataset
# NLP Assignment 1 | BITS WILP S2-25_AIMLCZG530
# =============================================================================
#
# !! UPDATE THE PATH BELOW to point to your downloaded AG News CSV file !!
DATA_PATH: str = "train.csv"
# =============================================================================

import collections
import logging
import os
import random
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1.  Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for full reproducibility across all libraries.

    Args:
        seed: Integer seed value applied globally to Python's ``random``
            module, NumPy, PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# =============================================================================
# 2.  Data Loading
# =============================================================================

def load_ag_news(filepath: str, n_articles: int = 5000) -> pd.DataFrame:
    """Load the AG News CSV and return the first *n_articles* rows.

    The function performs a **case-insensitive** column search for ``'text'``
    or ``'description'`` to handle the two common Kaggle formats of the
    AG News dataset.  If neither is found a descriptive ``ValueError`` is
    raised listing the available column names.

    Args:
        filepath: Absolute or relative path to the AG News ``.csv`` file.
        n_articles: Number of articles to subset (default: ``5000``).

    Returns:
        Single-column :class:`pandas.DataFrame` with column name ``'text'``
        containing raw article strings.

    Raises:
        FileNotFoundError: When the CSV file does not exist at *filepath*.
        ValueError: When no usable text column is found in the CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.  "
            "Download from https://www.kaggle.com/datasets/"
            "ilhamferdiona/ag-news-classification-dataset "
            "and update DATA_PATH at the top of this script."
        )

    logger.info("Loading dataset from '%s'", filepath)
    df = pd.read_csv(filepath, header=0)
    logger.info("Raw dataset shape: %s", df.shape)

    # Case-insensitive column resolution
    col_map: Dict[str, str] = {c.strip().lower(): c for c in df.columns}
    if "text" in col_map:
        text_col = col_map["text"]
    elif "description" in col_map:
        text_col = col_map["description"]
    else:
        raise ValueError(
            f"No suitable text column found.  "
            f"Available columns: {list(df.columns)}.  "
            "Expected 'Text' or 'Description'."
        )

    logger.info("Using column '%s' as text source", text_col)
    df = (
        df[[text_col]]
        .rename(columns={text_col: "text"})
        .head(n_articles)
        .copy()
    )
    df["text"] = df["text"].fillna("").astype(str)
    logger.info("Loaded %d articles", len(df))
    return df


# =============================================================================
# 3.  Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """Pipeline for cleaning and tokenising raw English text corpora.

    Applies the following steps **in order**:

    1. Lowercase conversion.
    2. Removal of digit sequences.
    3. Removal of punctuation and special characters.
    4. Whitespace normalisation (collapse runs to a single space).
    5. Word tokenisation via NLTK's ``word_tokenize``.
    6. Stopword removal using NLTK's English stopword list.
    7. Single-character token removal.

    Attributes:
        _stopwords: Frozen set of English stopwords from NLTK.
        _num_pattern: Compiled regex matching digit sequences.
        _punct_pattern: Compiled regex matching non-word, non-space chars.
        _space_pattern: Compiled regex matching whitespace runs.
    """

    def __init__(self) -> None:
        """Download required NLTK corpora and compile regex patterns."""
        logger.info("Initialising TextPreprocessor — downloading NLTK resources")
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        from nltk.corpus import stopwords as _sw
        self._stopwords: frozenset = frozenset(_sw.words("english"))
        self._num_pattern = re.compile(r"\d+")
        self._punct_pattern = re.compile(r"[^\w\s]")
        self._space_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        """Normalise a single raw text string.

        Args:
            text: Arbitrary raw input string.

        Returns:
            Lowercase string with digits, punctuation, and excess
            whitespace removed.
        """
        text = text.lower()
        text = self._num_pattern.sub(" ", text)
        text = self._punct_pattern.sub(" ", text)
        text = self._space_pattern.sub(" ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenise and filter a pre-cleaned text string.

        Args:
            text: A cleaned (lowercased, punctuation-free) string.

        Returns:
            List of word tokens with stopwords and single characters
            removed.
        """
        tokens = nltk.word_tokenize(text)
        return [
            t for t in tokens
            if t and len(t) > 1 and t not in self._stopwords
        ]

    def process_corpus(self, texts: pd.Series) -> List[List[str]]:
        """Apply the full preprocessing pipeline to a pandas Series.

        Args:
            texts: Series of raw article strings.

        Returns:
            List of token lists, one per article.
        """
        logger.info("Preprocessing %d articles …", len(texts))
        tokenized = [self.tokenize(self.clean(doc)) for doc in texts]
        total_tokens = sum(len(t) for t in tokenized)
        logger.info("Preprocessing complete — total tokens: %d", total_tokens)
        return tokenized


# =============================================================================
# 4.  Vocabulary
# =============================================================================

class Vocabulary:
    """Builds and manages a bidirectional word-index mapping.

    Tokens whose corpus frequency falls below *min_freq* are mapped to the
    ``<UNK>`` token.  Special tokens use fixed indices:

    * ``<PAD>`` → index **0** (embedding row kept zero; no gradient).
    * ``<UNK>`` → index **1**.

    After :meth:`build` the regular vocabulary tokens are inserted in
    **descending frequency order**, making ``word2idx`` order deterministic.

    Attributes:
        PAD_TOKEN: String constant ``'<PAD>'``.
        UNK_TOKEN: String constant ``'<UNK>'``.
        min_freq: Minimum inclusion frequency.
        word2idx: Mapping from token string to integer index.
        idx2word: Mapping from integer index to token string.
        word_freq: Raw :class:`collections.Counter` of all token occurrences.
    """

    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"

    def __init__(self, min_freq: int = 3) -> None:
        """Initialise an empty vocabulary container.

        Args:
            min_freq: Tokens occurring fewer than *min_freq* times are
                treated as out-of-vocabulary (default: ``3``).
        """
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: collections.Counter = collections.Counter()

    @property
    def vocab_size(self) -> int:
        """Total number of entries including special tokens."""
        return len(self.word2idx)

    def build(self, tokenized_corpus: List[List[str]]) -> None:
        """Construct the vocabulary from a tokenised corpus.

        Token insertion order after the special tokens is **descending
        frequency**, so the second most frequent word is always at index 2.

        Args:
            tokenized_corpus: List of token lists (one list per document).
        """
        logger.info("Building vocabulary with min_freq=%d …", self.min_freq)
        for tokens in tokenized_corpus:
            self.word_freq.update(tokens)

        # Reserve special-token slots
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

        # Descending-frequency ordering for determinism
        qualified = sorted(
            ((w, c) for w, c in self.word_freq.items() if c >= self.min_freq),
            key=lambda x: x[1],
            reverse=True,
        )
        for word, _ in qualified:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(
            "Vocabulary built — %d qualified tokens (min_freq=%d); "
            "total vocab size (incl. specials): %d",
            len(qualified),
            self.min_freq,
            self.vocab_size,
        )

    def encode(self, token: str) -> int:
        """Return the index for *token*, falling back to ``<UNK>``.

        Args:
            token: Input word string.

        Returns:
            Integer index into the embedding table.
        """
        return self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])

    def get_second_most_frequent_word(self) -> str:
        """Return the second most frequent word, excluding special tokens.

        Because :meth:`build` inserts tokens in descending frequency order,
        the qualified token list is already sorted; index 1 is the second
        most frequent.

        Returns:
            The second most frequent word string in the vocabulary.

        Raises:
            ValueError: If the vocabulary contains fewer than two qualified
                regular (non-special) tokens.
        """
        qualified = [
            w for w in self.word2idx
            if w not in (self.PAD_TOKEN, self.UNK_TOKEN)
        ]
        if len(qualified) < 2:
            raise ValueError(
                "Vocabulary too small — cannot retrieve second most frequent word."
            )
        # index 0 → most frequent, index 1 → second most frequent
        return qualified[1]


# =============================================================================
# 5.  Dataset & DataLoader
# =============================================================================

class NGramDataset(Dataset):
    """PyTorch Dataset yielding n-gram context–target pairs.

    For a context window of size *k* every valid position *i* in a token
    sequence produces one sample:

    * **context** — ``[w_i, w_{i+1}, …, w_{i+k-1}]``  (shape ``(k,)``)
    * **target**  — ``w_{i+k}``                         (scalar)

    Attributes:
        context_window: Number of preceding tokens per context vector.
        data: List of ``(context_indices, target_index)`` tuples.
    """

    def __init__(
        self,
        tokenized_corpus: List[List[str]],
        vocab: Vocabulary,
        context_window: int = 3,
    ) -> None:
        """Generate all context–target pairs from the tokenised corpus.

        Args:
            tokenized_corpus: List of token lists, one per document.
            vocab: A fully built :class:`Vocabulary` instance.
            context_window: Size of the context (default: ``3``).
        """
        self.context_window = context_window
        self.data: List[Tuple[List[int], int]] = []

        for tokens in tokenized_corpus:
            if len(tokens) <= context_window:
                continue
            encoded = [vocab.encode(t) for t in tokens]
            for i in range(len(encoded) - context_window):
                context = encoded[i : i + context_window]
                target = encoded[i + context_window]
                self.data.append((context, target))

        logger.info(
            "NGramDataset created — %d context–target pairs (context_window=%d)",
            len(self.data),
            context_window,
        )

    def __len__(self) -> int:
        """Return total number of (context, target) samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one (context, target) pair.

        Args:
            idx: Integer sample index.

        Returns:
            Tuple of:
                * context — :class:`torch.LongTensor` of shape
                  ``(context_window,)``
                * target  — scalar :class:`torch.LongTensor`
        """
        context, target = self.data[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def build_dataloader(
    dataset: NGramDataset,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap an :class:`NGramDataset` in a :class:`torch.utils.data.DataLoader`.

    Args:
        dataset: Constructed :class:`NGramDataset` instance.
        batch_size: Mini-batch size (default: ``512``).
        shuffle: Whether to shuffle samples each epoch (default: ``True``).

    Returns:
        Configured :class:`DataLoader` instance.
    """
    loader: DataLoader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    logger.info(
        "DataLoader created — %d batches of up to %d samples",
        len(loader),
        batch_size,
    )
    return loader


# =============================================================================
# 6.  Model Architecture
# =============================================================================

class FeedForwardNLM(nn.Module):
    """Feed-Forward Neural Language Model implemented in PyTorch.

    Architecture (forward pass)::

        Input (B, C)  →  Embedding  →  (B, C, E)
                      →  Flatten    →  (B, C×E)
                      →  Linear     →  (B, H)
                      →  ReLU       →  (B, H)
                      →  Linear     →  (B, V)   ← raw logits

    where B = batch size, C = context window, E = embedding dim,
    H = hidden dim, V = vocabulary size.

    Attributes:
        embedding: Learnable embedding table of shape ``(V, E)``.
        fc1: Hidden fully-connected layer ``(C×E → H)``.
        relu: ReLU activation.
        fc2: Output projection layer ``(H → V)``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        context_window: int = 3,
        padding_idx: int = 0,
    ) -> None:
        """Initialise the Feed-Forward NLM.

        Args:
            vocab_size: Total vocabulary size; determines embedding table
                height and output layer width.
            embed_dim: Word embedding dimensionality (default: ``300``).
            hidden_dim: Number of neurons in the hidden layer (default:
                ``128``).
            context_window: Number of context tokens per sample (default:
                ``3``).
            padding_idx: Index of ``<PAD>``; its gradient is zeroed
                (default: ``0``).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc1 = nn.Linear(context_window * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        logger.info(
            "FeedForwardNLM initialised — vocab=%d | embed_dim=%d | "
            "hidden_dim=%d | context_window=%d | input_dim=%d",
            vocab_size,
            embed_dim,
            hidden_dim,
            context_window,
            context_window * embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            x: :class:`torch.LongTensor` of shape ``(B, context_window)``
               containing token indices.

        Returns:
            :class:`torch.FloatTensor` of shape ``(B, vocab_size)``
            containing raw (unnormalised) logits.
        """
        embedded = self.embedding(x)                     # (B, C, E)
        flattened = embedded.view(embedded.size(0), -1)  # (B, C*E)
        hidden = self.relu(self.fc1(flattened))          # (B, H)
        logits = self.fc2(hidden)                        # (B, V)
        return logits


# =============================================================================
# 7.  Trainer
# =============================================================================

class Trainer:
    """Encapsulates the supervised training loop for :class:`FeedForwardNLM`.

    Uses the **Adam** optimiser and **Cross-Entropy** loss.  Per-epoch mean
    loss is recorded and returned for downstream visualisation.

    Attributes:
        model: Neural language model being trained.
        dataloader: DataLoader providing (context, target) mini-batches.
        device: Torch device on which tensors and model reside.
        optimizer: Adam optimiser instance.
        criterion: Cross-entropy loss function.
        epoch_losses: Populated with mean epoch loss after :meth:`train`.
    """

    def __init__(
        self,
        model: FeedForwardNLM,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
    ) -> None:
        """Initialise the Trainer.

        Args:
            model: :class:`FeedForwardNLM` to optimise.
            dataloader: Training DataLoader.
            device: Target device (CPU or CUDA).
            lr: Learning rate for Adam (default: ``0.001``).
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_losses: List[float] = []
        logger.info("Trainer initialised — device=%s | lr=%g", device, lr)

    def train(self, n_epochs: int = 10) -> List[float]:
        """Execute the training loop for *n_epochs* passes over the data.

        Args:
            n_epochs: Number of training epochs (default: ``10``).

        Returns:
            List of mean cross-entropy loss values, one per epoch.
        """
        logger.info("Starting training — %d epochs", n_epochs)
        self.model.train()

        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            n_batches = 0

            for contexts, targets in self.dataloader:
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(contexts)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            mean_loss = running_loss / n_batches
            self.epoch_losses.append(mean_loss)
            logger.info(
                "Epoch [%02d/%02d] — Mean Loss: %.6f", epoch, n_epochs, mean_loss
            )

        final_loss = self.epoch_losses[-1]
        print(f"\n{'=' * 60}")
        print(f"  Training Complete")
        print(f"  Final Loss (Epoch {n_epochs}): {final_loss:.6f}")
        print(f"{'=' * 60}\n")
        return self.epoch_losses


# =============================================================================
# 8.  Visualization
# =============================================================================

def plot_training_loss(
    losses: List[float],
    save_path: str = "training_loss_curve.png",
) -> None:
    """Plot the per-epoch training loss and save to disk.

    Produces a Seaborn line-plot with markers, labelled axes, a title,
    and a grid.  The figure is saved as a PNG before being displayed.

    Args:
        losses: List of mean loss values, one entry per epoch.
        save_path: Output file path for the PNG (default:
            ``'training_loss_curve.png'``).
    """
    sns.set_theme(style="darkgrid", palette="muted")
    epochs = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        x=epochs,
        y=losses,
        ax=ax,
        marker="o",
        linewidth=2.2,
        color="#2196F3",
        markersize=7,
        label="Training Loss",
    )
    ax.set_title(
        "Feed-Forward Neural Language Model — Training Loss Curve",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Training loss curve saved to '%s'", save_path)
    plt.show()


# =============================================================================
# 9.  Theoretical Analysis
# =============================================================================

THEORETICAL_ANALYSIS: str = """
╔══════════════════════════════════════════════════════════════════════════╗
║        THEORETICAL ANALYSIS — COUNT-BASED vs. NEURAL LM EMBEDDINGS     ║
╚══════════════════════════════════════════════════════════════════════════╝

Count-based (N-Gram) methods construct word representations by tallying
co-occurrence statistics within a fixed context window, producing sparse,
high-dimensional vectors (e.g., PPMI matrices or TF-IDF variants).  These
vectors are directly interpretable and inexpensive to compute, but they
capture only shallow, surface-level co-occurrence patterns; two words that
rarely appear in identical contexts yield nearly orthogonal vectors even when
they are semantically synonymous (e.g., "automobile" vs. "car").  Because
every word's representation is derived solely from its own observed counts,
rare words accumulate insufficient statistical evidence to form reliable
embeddings — their vectors are noisy, underspecified, and fail to encode
meaningful semantic neighbourhoods.

Prediction-based (Neural LM) methods, exemplified by the Feed-Forward model
implemented here, learn dense, low-dimensional embeddings by optimising a
predictive objective: the model must correctly forecast a target word from its
surrounding context.  During back-propagation the gradient signal updates each
word's embedding by propagating information from semantically related contexts,
enabling generalisation across words that share distributional properties.
Crucially, this architecture handles rare words more gracefully than
count-based methods: through shared hidden-layer parameters, a rare word whose
few occurrences co-occur with common, well-trained words can still acquire a
meaningful dense vector via indirect gradient flow.  Furthermore, the
continuous embedding space encodes graded analogical and synonym relationships
that co-occurrence matrices cannot represent (e.g., king − man + woman ≈ queen).

Verdict: Prediction-based Neural LM embeddings substantially outperform
count-based N-Gram embeddings for rare words, because the neural objective
function allows gradient information to flow from high-frequency neighbours
into rare-word representations, yielding denser, semantically richer vectors
even from limited training signal.
"""


# =============================================================================
# 10. Main Orchestration
# =============================================================================

def main() -> None:
    """End-to-end pipeline for the Feed-Forward Neural Language Model.

    Execution steps:

    1. Fix all random seeds for reproducibility.
    2. Detect compute device (CUDA GPU if available, else CPU).
    3. Load and subset the AG News dataset (first 5 000 articles).
    4. Clean, tokenise, and remove stopwords from the corpus.
    5. Build vocabulary with ``min_freq=3``; ``<PAD>``→0, ``<UNK>``→1.
    6. Generate n-gram context–target pairs (``context_window=3``).
    7. Construct a :class:`FeedForwardNLM` and move it to the target device.
    8. Train for 10 epochs with Cross-Entropy loss; print final loss.
    9. Display the embedding vector for the second most frequent word.
    10. Save and display the training loss curve as a PNG.
    11. Print the theoretical analysis comparing N-Gram and Neural LM
        embeddings.
    """
    # ── 1. Reproducibility ────────────────────────────────────────────────
    set_seed(42)

    # ── 2. Device detection ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Compute device: %s", device)

    # ── 3. Data loading ───────────────────────────────────────────────────
    df = load_ag_news(DATA_PATH, n_articles=5000)

    # ── 4. Text preprocessing ─────────────────────────────────────────────
    preprocessor = TextPreprocessor()
    tokenized_corpus = preprocessor.process_corpus(df["text"])

    # ── 5. Vocabulary ─────────────────────────────────────────────────────
    vocab = Vocabulary(min_freq=3)
    vocab.build(tokenized_corpus)
    logger.info("Final vocabulary size: %d", vocab.vocab_size)

    # ── 6. Dataset & DataLoader ───────────────────────────────────────────
    dataset = NGramDataset(tokenized_corpus, vocab, context_window=3)
    if len(dataset) == 0:
        logger.error(
            "Dataset is empty — verify that DATA_PATH points to a valid "
            "AG News CSV file and that the file is non-empty."
        )
        sys.exit(1)
    dataloader = build_dataloader(dataset, batch_size=512, shuffle=True)

    # ── 7. Model ──────────────────────────────────────────────────────────
    model = FeedForwardNLM(
        vocab_size=vocab.vocab_size,
        embed_dim=300,
        hidden_dim=128,
        context_window=3,
        padding_idx=vocab.word2idx[Vocabulary.PAD_TOKEN],
    )

    # ── 8. Training ───────────────────────────────────────────────────────
    trainer = Trainer(model, dataloader, device, lr=1e-3)
    epoch_losses = trainer.train(n_epochs=10)

    # ── 9. Embedding for second most frequent word ────────────────────────
    second_word: str = vocab.get_second_most_frequent_word()
    second_word_idx: int = vocab.encode(second_word)
    embedding_vector: np.ndarray = (
        model.embedding.weight[second_word_idx].detach().cpu().numpy()
    )

    print(f"{'─' * 60}")
    print(f"  Second Most Frequent Word  : '{second_word}'")
    print(f"  Corpus Frequency           : {vocab.word_freq[second_word]}")
    print(f"  Vocabulary Index           : {second_word_idx}")
    print(f"  Embedding Vector Shape     : {embedding_vector.shape}")
    print(f"  First 10 Dimensions        : {np.round(embedding_vector[:10], 6)}")
    print(f"\n  Full Embedding Vector ({embedding_vector.shape[0]}D):")
    print(f"  {embedding_vector}")
    print(f"{'─' * 60}\n")

    # ── 10. Training loss curve ───────────────────────────────────────────
    plot_training_loss(epoch_losses, save_path="training_loss_curve.png")

    # ── 11. Theoretical analysis ──────────────────────────────────────────
    print(THEORETICAL_ANALYSIS)


if __name__ == "__main__":
    main()
