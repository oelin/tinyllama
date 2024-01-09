from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


class RMSNorm(nn.Module):
    """Root Mean Squared (RMS) normalization.

    Example
    -------
    >>> module = RMSNorm()
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self) -> None:
        """Initialize the module."""

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True))

        return x / rms


class SwiGLU(nn.Module):
    """Swish GLU (SwiGLU).

    Implements the SwiGLU activation function (Shazeer et al., 2020). The
    nonlinearity combines Swish and GLU. Note that this implementation is
    somewhat simplified and uses SiLU (Swish-1).

    Example
    -------
    >>> module = SwiGLU(embedding_dimension=256, gated=True)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, *, embedding_dimension: int, gated: bool) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        gated : int
            Specifies whether to use gating.
        """

        super().__init__()

        if gated:
            self.linear = nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension,
            )

            nn.init.ones_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

            self.activation = lambda x: F.silu(x) * self.linear(x)
        else:
            self.activation = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        return self.activation(x)


class RoPE(nn.Module):
    """Rotary positional embedding (RoPE).

    Rotary positional embedding (Su et al., 2023) rotates keys and queries by
    their absolute position such that their dot product depends only on their
    content and *relative position*. Generalized to arbitrary dimensions, RoPE
    divides a D-dimensional space into D//2 subspaces.

    Example
    -------
    >>> module = RoPE(embedding_dimension=256, base=10_000)
    >>> q = torch.randn((1, 10, 256))
    >>> k = torch.randn((1, 10, 256))
    >>> alignment = torch.einsum('bte,bse->bts', module(q), module(k))
    """

    def __init__(self, *, embedding_dimension: int, base: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        base : int
            The base to use for absolute positional encodings.
        """

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.base = base

        # Precompute theta.

        exponent = torch.arange(
            start=0,
            end=embedding_dimension,
            step=2,
            dtype=torch.float,
        ) / embedding_dimension

        theta = 1. / torch.pow(base, exponent)

        self.theta = theta

    def absolute_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Perform absolute positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        encoding : torch.Tensor
            The absolute positional encoding.
        """

        if self.theta.device != x.device:
            self.theta = self.theta.to(x.device)

        encoding = torch.einsum(
            't,e->te',
            torch.arange(x.size(1), dtype=torch.float, device=x.device),
            self.theta,
        )

        encoding = repeat(encoding, '... e -> ... (e n)', n=2)

        return encoding

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate each subspace by -90 degrees."""

        x = rearrange(x, '... (e n) -> ... e n', n=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = rearrange(x, '... e n -> ... (e n)')

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Foward pass."""

        encoding = self.absolute_positional_encoding(x)
        x = x * encoding.cos() + (self.rotate_half(x) * encoding.sin())

        return x


class GQA(nn.Module):
    """GQA.

    Implements Grouped Query Attention (GQA) (Ainsile et al., 2023). GQA shares
    groups of keys and values across heads in order to reduce the size of key
    and value projection matricies. It generalizes Multi-query Attention (MQA).

    Example
    -------
    >>> module = GQA(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ...     number_of_groups=4,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_groups: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of heads.
        number_of_groups : int
            The number of groups.
        """

        super().__init__()

        self.number_of_heads = number_of_heads
        self.number_of_groups = number_of_groups

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=2 * (embedding_dimension // number_of_heads) * number_of_groups,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_3 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        h = self.number_of_heads
        g = self.number_of_groups

        k, v = rearrange(self.linear_1(x), 'b s (n g e) -> n b g s e', n=2, g=g)
        q = rearrange(self.linear_2(x), 'b t (h e) -> b h t e', h=h)

        score = torch.einsum('bhte,bgse->bhts', q, k)
        score = score + mask if mask is not None else score
        score = F.softmax(score, dim=-1)

        x = torch.einsum('bhts,bgse->bthe', score, v)
        x = self.linear_3(rearrange(x, 'b t h e -> b t (h e)'))

        return x


class MLP(nn.Module):
    """MLP.

    Implements the shallow MLP used in transformer blocks.

    Example
    -------
    >>> module = MLP(embedding_dimension=256, intermediate_dimension=256)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        intermediate_dimension: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        intermediate_dimension : int
            The intermediate dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension,
                out_features=intermediate_dimension,
            ),
            SwiGLU(embedding_dimension=intermediate_dimension, gated=False),
            nn.Linear(
                in_features=intermediate_dimension,
                out_features=embedding_dimension,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        return self.layers(x)


class FactoredEmbedding(nn.Module):
    """Factored embedding.

    Implements a factored embedding layer. Rather than projecting directly from
    tokens to embeddings (vocabulary_size * embedding_dimension), we can first
    map to a lower dimension, and then back up to the embedding dimension. This
    reduces the number of embedding parameters by several orders of magnitude.

    Example
    -------
    >>> module = FactoredEmbedding(
    ...     vocabulary_size=32_000,
    ...     embedding_dimension=256,
    ...     factor_dimension=16,
    ... )
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_dimension: int,
        factor_dimension: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        vocabulary_size : int
            The vocabulary size.
        embedding_dimension : int
            The embedding dimension.
        factor_dimension : int
            The factor dimension, should be less than `embedding_dimension`.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Embedding(
                num_embeddings=vocabulary_size,
                embedding_dim=factor_dimension,
            ),
            nn.Linear(
                in_features=factor_dimension,
                out_features=embedding_dimension,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. A sequence of token ids.

        Returns
        -------
        embedding : torch.Tensor
            The output tensor. Token embeddings.
        """

        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer block.

    Implements a transformer (decoder) block (Radford et al., 2019).

    Example
    -------
    >>> module = TransformerBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        intermediate_dimension: int,
        number_of_heads: int,
        number_of_groups: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        intermediate_dimension : int
            The intermdiate dimension of the MLP.
        number_of_heads : int
            The number of GQA heads.
        number_of_groups : int
            The number of GQA groups.
        """

        super().__init__()

        self.attention = GQA(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_groups=number_of_groups,
        )

        self.mlp = MLP(
            embedding_dimension=embedding_dimension,
            intermediate_dimension=intermediate_dimension,
        )

        self.normalization_1 = RMSNorm()
        self.normalization_2 = RMSNorm()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.attention(self.normalization_1(x), mask=mask)
        x = x + self.mlp(self.normalization_2(x))

        return x


@dataclass(frozen=True)
class TinyLlamaConfiguration:
    embedding_dimension: int = 256
    intermediate_dimension: int = 704
    number_of_heads: int = 16
    number_of_layers: int = 16
    vocabulary_size: int = 1024
    context_length: int = 1024


# TODO: implement RoPE and KV caches.

class TinyLlama(nn.Module):
    """TinyLlama.

    Implements TinyLlama (Zhang et al., 2024). TinyLlama is LLaMA-style
    transformer architecture with 1.1B parameters, that demonstrates superior
    performance to contemporary open-source language models of a similar size.

    As discussed in the paper, TinyLlama employs modern modules such as:

    - RoPE,
    - RMSNorm,
    - SwiGLU,
    - GQA.

    In addition, the original implementation integrates Flash Attention 2, and
    fused modules to optimize throughput. This implementatoin will accomodate
    these features in future.

    Example
    -------
    >>> # As specified in Table 1 of (Zhang et al., 2024).
    >>> configuration = TinyLlamaConfiguration(
    ...     embedding_dimension=2048,
    ...     intermediate_dimension=5632,  # x2.75.
    ...     number_of_heads=16,
    ...     number_of_layers=22,
    ...     vocabulary_size=32_000,
    ...     context_length=2048,
    ... )
    >>> model = TinyLlama(configuration=configuration)
    >>> tokens = tokenizer.encode('Hello, World!')
    >>> logits = model(tokens, mask=None)
    """

    def __init__(self, *, configuration: TinyLlamaConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : TinyLlamaConfiguration
            The model configuration.
        """

        super().__init__()

        self.configuration = configuration

        factor_dimension = 64

        self.embedding = FactoredEmbedding(
            vocabulary_size=configuration.vocabulary_size,
            embedding_dimension=configuration.embedding_dimension,
            factor_dimension=factor_dimension,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimension=configuration.embedding_dimension,
                intermediate_dimension=configuration.intermediate_dimension,
                number_of_heads=configuration.number_of_heads,
                number_of_groups=configuration.number_of_heads // 4,
            ) for _ in range(configuration.number_of_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. A sequence of token ids.
        mask : Optional[torch.Tensor]
            A attention mask such as a causal mask for autogressive decoding.

        Returns
        -------
        logits : torch.Tensor
            The unnormalized logit distribution at each position.
        """

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        # Parameter tied unembedding.

        x = x @ self.embedding.layers[1].weight
        x = x @ self.embedding.layers[0].weight.T

        logits = x

        return logits
