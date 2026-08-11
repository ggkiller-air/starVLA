"""Tactile fusion and JEPA primitives shared by the GR00T-style heads."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

RAW_DIM = 768
REGION_GRIDS = ((6, 8), (5, 8), (2, 4), (1, 4), (2, 4), (1, 4), (16, 16), (16, 16))
REGION_SIZES = tuple(rows * cols for rows, cols in REGION_GRIDS)

# Region-major, zero-based channel mapping from the Unitree G1 SONIC skin spec.
_VALID_IDX_ONE_BASED = (
    195,
    211,
    227,
    243,
    3,
    19,
    35,
    51,
    196,
    212,
    228,
    244,
    4,
    20,
    36,
    52,
    197,
    213,
    229,
    245,
    5,
    21,
    37,
    53,
    198,
    214,
    230,
    246,
    6,
    22,
    38,
    54,
    199,
    215,
    231,
    247,
    7,
    23,
    39,
    55,
    200,
    216,
    232,
    248,
    8,
    24,
    40,
    56,
    58,
    42,
    26,
    10,
    250,
    234,
    218,
    202,
    59,
    43,
    27,
    11,
    251,
    235,
    219,
    203,
    60,
    44,
    28,
    12,
    252,
    236,
    220,
    204,
    61,
    45,
    29,
    13,
    253,
    237,
    221,
    205,
    62,
    46,
    30,
    14,
    254,
    238,
    222,
    206,
    79,
    95,
    111,
    127,
    80,
    96,
    112,
    128,
    9,
    25,
    41,
    57,
    177,
    162,
    146,
    130,
    178,
    161,
    145,
    129,
    249,
    233,
    217,
    201,
)
_VEST_VALID_IDX = tuple(index - 1 for index in _VALID_IDX_ONE_BASED)
_ARM_ORDER = tuple(range(128, 256)) + tuple(range(128))
VALID_IDX = _VEST_VALID_IDX + tuple(256 + i for i in _ARM_ORDER) + tuple(512 + i for i in _ARM_ORDER)


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class PerRegionMLP(nn.Module):
    def __init__(self, region_sizes: tuple[int, ...], hidden_dim: int, embed_dim: int):
        super().__init__()
        self.region_sizes = region_sizes
        self.branches = nn.ModuleList(TwoLayerMLP(size, hidden_dim, embed_dim) for size in region_sizes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        parts = torch.split(inputs, self.region_sizes, dim=-1)
        return torch.stack([branch(part) for branch, part in zip(self.branches, parts, strict=True)], dim=1)


class PerRegionCNN(nn.Module):
    def __init__(
        self,
        region_grids: tuple[tuple[int, int], ...],
        embed_dim: int,
        channels: int,
        pool: tuple[int, int],
        coord: bool,
        coord_scale: float,
    ):
        super().__init__()
        self.region_grids = region_grids
        self.coord = coord
        self.coord_scale = coord_scale
        input_channels = 3 if coord else 1
        self.convs = nn.ModuleList()
        self.projections = nn.ModuleList()
        for rows, cols in region_grids:
            pooled_rows = min(pool[0], rows)
            pooled_cols = min(pool[1], cols)
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((pooled_rows, pooled_cols)),
                )
            )
            self.projections.append(nn.Linear(channels * pooled_rows * pooled_cols, embed_dim))

    @staticmethod
    def _coordinate(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if size == 1:
            return torch.zeros(1, device=device, dtype=dtype)
        return torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sizes = tuple(rows * cols for rows, cols in self.region_grids)
        parts = torch.split(inputs, sizes, dim=-1)
        tokens = []
        for (rows, cols), conv, projection, part in zip(
            self.region_grids, self.convs, self.projections, parts, strict=True
        ):
            grid = part.reshape(part.shape[0], 1, rows, cols)
            if self.coord:
                row = self._coordinate(rows, grid.device, grid.dtype) * self.coord_scale
                col = self._coordinate(cols, grid.device, grid.dtype) * self.coord_scale
                row = row.view(1, 1, rows, 1).expand(grid.shape[0], 1, rows, cols)
                col = col.view(1, 1, 1, cols).expand(grid.shape[0], 1, rows, cols)
                grid = torch.cat((grid, row, col), dim=1)
            tokens.append(projection(conv(grid).flatten(1)))
        return torch.stack(tokens, dim=1)


class TactileSlotAggregator(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int, num_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, region_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(region_tokens.shape[0], -1, -1)
        queries = queries.to(region_tokens.dtype)
        # The 192-wide heads used by DiT-L are unstable in bf16 flash backward.
        with sdpa_kernel([SDPBackend.MATH]):
            output, _ = self.attention(queries, region_tokens, region_tokens, need_weights=False)
        return self.norm(output)


class TactileEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_tokens: int = 8,
        num_heads: int = 8,
        encoder_type: str = "mlp",
        cnn_channels: int = 32,
        cnn_pool: tuple[int, int] = (2, 2),
        coord_scale: float = 0.1,
        raw_dim: int = RAW_DIM,
        valid_idx: tuple[int, ...] = VALID_IDX,
        region_grids: tuple[tuple[int, int], ...] = REGION_GRIDS,
    ):
        super().__init__()
        valid_idx = tuple(int(index) for index in valid_idx)
        region_grids = tuple((int(rows), int(cols)) for rows, cols in region_grids)
        region_sizes = tuple(rows * cols for rows, cols in region_grids)
        if sum(region_sizes) != len(valid_idx):
            raise ValueError("Tactile region grids do not cover all valid channels")
        if not valid_idx or min(valid_idx) < 0 or max(valid_idx) >= raw_dim:
            raise ValueError("Tactile valid-channel indices are outside raw packet bounds")
        if len(set(valid_idx)) != len(valid_idx):
            raise ValueError("Tactile valid-channel indices must be unique")
        self.raw_dim = int(raw_dim)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.register_buffer("valid_idx", torch.tensor(valid_idx, dtype=torch.long), persistent=False)
        if encoder_type == "mlp":
            self.per_region = PerRegionMLP(region_sizes, hidden_dim, embed_dim)
        elif encoder_type in {"cnn", "coord"}:
            self.per_region = PerRegionCNN(
                region_grids,
                embed_dim,
                channels=cnn_channels,
                pool=cnn_pool,
                coord=encoder_type == "coord",
                coord_scale=coord_scale,
            )
        else:
            raise ValueError(f"Unknown tactile encoder type: {encoder_type!r}")
        self.aggregator = TactileSlotAggregator(embed_dim, num_tokens, num_heads)

    def select_and_normalize(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape[-1] != self.raw_dim:
            raise ValueError(f"Expected tactile width {self.raw_dim}, got {raw.shape}")
        return raw.index_select(-1, self.valid_idx).to(self.aggregator.norm.weight.dtype) / 255.0

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.per_region(self.select_and_normalize(raw)))

    def encode_pooled(self, raw: torch.Tensor) -> torch.Tensor:
        has_time = raw.ndim == 3
        if not has_time:
            raw = raw.unsqueeze(1)
        batch, time = raw.shape[:2]
        tokens = self.forward(raw.reshape(batch * time, raw.shape[-1]))
        pooled = tokens.mean(dim=1).reshape(batch, time, self.embed_dim)
        return pooled if has_time else pooled[:, 0]


class DreamHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, horizon: int, hidden_dim: int):
        super().__init__()
        self.horizon = horizon
        self.target_dim = target_dim
        self.net = TwoLayerMLP(input_dim, hidden_dim, horizon * target_dim)

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        output = self.net(trunk)
        return output.reshape(output.shape[0], self.horizon, self.target_dim)


def jepa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.detach().float()
    direction = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    magnitude = F.smooth_l1_loss(prediction.norm(dim=-1), target.norm(dim=-1), reduction="none")
    loss = direction + beta * magnitude
    if mask is None:
        return loss.mean()
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    if mask.shape != loss.shape:
        raise ValueError(f"JEPA mask shape {tuple(mask.shape)} does not match loss {tuple(loss.shape)}")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def build_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    teacher_parameters = tuple(teacher.parameters())
    student_parameters = tuple(student.parameters())
    if len(teacher_parameters) != len(student_parameters):
        raise RuntimeError("EMA teacher and student parameter structures differ")
    for target, online in zip(teacher_parameters, student_parameters, strict=True):
        target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)
    for target, online in zip(teacher.buffers(), student.buffers(), strict=True):
        target.copy_(online)
