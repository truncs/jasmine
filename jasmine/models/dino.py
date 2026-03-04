import re
import functools
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as onp
import torch


DINOV3_S_URL = 'https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNnRjenRmNWI2eWhxeGNnZ2lkeXo5cThyIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTc3OTl9fX1dfQ__&Signature=SgxoGnoHPC3%7EawZUdsdBY-gdSLsiqWRJcDnAk0dPGsYOjc-aFycl5wIVo645aomnqAc7VOTuobVUx6thud1noEGDwtlbRnQvNMRx2cbkZVGnzhM-F9mZWCpoAAfbrIH6FBl8Tpa7Z77xxafyXMH4S8BzPrp0dgY1tqJJzVkLtH8e2N%7E%7EBbwFJkOGwZag06Q4ot0CfUxRPz3jtBj7jDbQVnRu7cwAckM6-i1rAODokA1IwolpXHToEPFbtLHWHO%7EyEpyWpvu9RT4QlN461hSxFD9nl-NyBsV%7EOju4w2BCJCYjjo2C1s3eH6PjqUp7uZN6xrvGWtHuN1wCrTFnA5vL9Q__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1312754137084299'


def load_vit_params(params_jax: dict, vit_pt: torch.nn.Module):
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(params_jax)
    dinov2_params = {path: param for path, param in vit_pt.named_parameters()}

    no_transpose = {
        "cls_token",
        "pos_embed",
        "mask_token",
        "storage_tokens",
    }
    dinov2_params_flat = []
    for path, param in jax_params_flat:
        shape = param.shape if hasattr(param, 'shape') else None
        original_path = ".".join([str(p.key) if hasattr(p, 'key') else str(p.name) if hasattr(p, 'name') else str(p.idx) if hasattr(p, 'idx') else str(p) for p in path])
        path_str = original_path
        if path_str.endswith('.value'):
            path_str = path_str[:-6]
        path_str = re.sub(r"\.scale|\.kernel", ".weight", path_str)
        
        if path_str in dinov2_params:
            dinov2_param = dinov2_params[path_str]
            if path_str not in no_transpose:
                if len(shape) == 4:
                    dinov2_param = torch.permute(dinov2_param, (2, 3, 1, 0))
                else:
                    dinov2_param = torch.permute(
                        dinov2_param, tuple(reversed(range(len(shape))))
                    )
            if shape != dinov2_param.shape:
                print("Shape mismatch:", path_str, shape, dinov2_param.shape)
            dinov2_params_flat.append(jnp.asarray(dinov2_param.detach().numpy()))
            dinov2_params.pop(path_str)
        elif "rngs" in original_path or not hasattr(param, 'shape'):
            dinov2_params_flat.append(param)
        else:
            print("JAX param not mapped to Torch:", original_path, "->", path_str, shape)
            dinov2_params_flat.append(param)
            
    for path, param in dinov2_params.items():
        print("Unmapped Torch param:", path, param.shape)

    return jax.tree_util.tree_unflatten(jax_param_pytree, dinov2_params_flat)


def rope_rotate_half(x: jnp.array) -> jnp.array:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concat([-x2, x1], axis=-1)


def rope_apply(x: jnp.array, sin: jnp.array, cos: jnp.array) -> jnp.array:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class Attention(nnx.Module):
    def __init__(self, embed_dim: int = 384, num_heads: int = 8,
                 attn_bias: bool = True, attn_drop_rate: float = 0.0,
                 proj_bias: bool = True, proj_drop_rate: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.attn_bias = attn_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_bias = proj_bias
        self.proj_drop_rate = proj_drop_rate
        self.embed_dim = embed_dim

        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=attn_bias, rngs=rngs)
        self.attn_drop = nnx.Dropout(rate=attn_drop_rate, rngs=rngs)
        self.proj = nnx.Linear(embed_dim, embed_dim, use_bias=proj_bias, rngs=rngs)
        self.proj_drop = nnx.Dropout(rate=proj_drop_rate, rngs=rngs)

    def _apply_rope(self, q, k, rope):
        # All operations will use the dtype of rope,
        # the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = jnp.astype(q, rope_dtype)
        k = jnp.astype(k, rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = jnp.concat((q_prefix, q), axis=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = jnp.concat((k_prefix, k), axis=-2)  # [B, head, N, D//head]
        q = jnp.astype(q, q_dtype)
        k = jnp.astype(k, k_dtype)
        return q, k
        
    def __call__(self, x, rope=None, training: bool = False):
        B, N, C = x.shape
        assert (
            C == self.embed_dim
        ), f"Input embedding dimension ({C}) should match layer embedding dimension ({self.embed_dim})."
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)

        if rope is not None:
            q, k = self._apply_rope(q, k, rope)

        # Attention matrix: (B, H, N, N)
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(C // self.num_heads)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, deterministic=not training)

        # Output: (B, N, H, C // H)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not training)

        return x


class LayerScale(nnx.Module):
    def __init__(self, dim: int, initial_value: float = 1.0, *, rngs: nnx.Rngs):
        self.initial_value = initial_value
        self.gamma = nnx.Param(initial_value * jnp.ones((dim,)))

    def __call__(self, x):
        return x * self.gamma.value


class DropPath(nnx.Module):
    def __init__(self, rate: float = 0.0, *, rngs: nnx.Rngs):
        self.rate = rate
        self.rngs = rngs

    def __call__(self, x, deterministic: bool = False):
        if self.rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = jax.random.bernoulli(
                self.rngs.dropout(), keep_prob, shape=shape
            )
            return x / keep_prob * random_tensor
        else:
            return x


class Mlp(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int = 1536, out_features: int = 384, act_layer=jax.nn.gelu, dropout_rate: float = 0.0, bias: bool = True, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.dropout_rate = dropout_rate
        self.bias = bias

        self.fc1 = nnx.Linear(in_features, hidden_features, use_bias=bias, rngs=rngs)
        self.drop1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, use_bias=bias, rngs=rngs)
        self.drop2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, training: bool = False):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop1(x, deterministic=not training)
        x = self.fc2(x)
        x = self.drop2(x, deterministic=not training)
        return x


class PatchEmbed(nnx.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 384, img_size: int = 224, patch_size: int = 14, norm_layer=None, flatten_embedding: bool = True, *, rngs: nnx.Rngs):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs
        )
        self.norm = norm_layer(embed_dim, rngs=rngs) if norm_layer is not None else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, H, W, C = x.shape
        patch_H, patch_W = self.patch_size, self.patch_size
        assert (
            H % patch_H == 0 and W % patch_W == 0
        ), f"Image size ({H}*{W}) cannot be evenly divided by patch size ({patch_H}*{patch_W})."

        x = self.proj(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm is not None:
            x = self.norm(x)

        if not self.flatten_embedding:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x


class Block(nnx.Module):
    def __init__(self, embed_dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0, drop_path_rate: float = 0.0, AttentionClass=Attention, FfnClass=Mlp, *, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate

        self.norm1 = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.attn = AttentionClass(embed_dim=embed_dim, num_heads=num_heads, rngs=rngs)
        self.ls1 = LayerScale(embed_dim, rngs=rngs)
        self.drop_path1 = DropPath(drop_path_rate, rngs=rngs)

        self.norm2 = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.mlp = FfnClass(in_features=embed_dim, hidden_features=int(mlp_ratio * embed_dim), out_features=embed_dim, rngs=rngs)
        self.ls2 = LayerScale(embed_dim, rngs=rngs)
        self.drop_path2 = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, rope: jnp.ndarray = None,
                 training: bool = False) -> jnp.ndarray:

        def attn_residual_func(x: jnp.ndarray, rope: jnp.ndarray) -> jnp.ndarray:
            x = self.norm1(x)
            x = self.attn(x, rope=rope, training=training)
            x = self.ls1(x)
            return x

        def ffn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = self.norm2(x)
            x = self.mlp(x, training=training)
            x = self.ls2(x)
            return x

        if training:
            x = x + self.drop_path1(attn_residual_func(x, rope), deterministic=not training)
            x = x + self.drop_path2(ffn_residual_func(x), deterministic=not training)
        else:
            x = x + attn_residual_func(x, rope)
            x = x + ffn_residual_func(x)

        return x


class DinoViT(nnx.Module):
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_pos_tokens: int = 1369,
        num_cls_tokens: int = 1,
        num_storage_tokens: int = 4,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        BlockClass=Block,
        AttentionClass=Attention,
        FfnClass=Mlp,
        EmbedLayer=PatchEmbed,
        *,
        rngs: nnx.Rngs,
    ):
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_pos_tokens = num_pos_tokens
        self.num_cls_tokens = num_cls_tokens
        self.num_storage_tokens = num_storage_tokens
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate

        self.patch_embed = EmbedLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            rngs=rngs,
        )
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))

        if self.num_storage_tokens > 0:
            self.storage_tokens = nnx.Param(jnp.zeros((1, num_storage_tokens, embed_dim)))

        blocks = []
        for i in range(depth):
            blocks.append(
                BlockClass(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path_rate=drop_path_rate,
                    AttentionClass=AttentionClass,
                    FfnClass=FfnClass,
                    rngs=rngs,
                )
            )
        self.blocks = nnx.List(blocks)
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def _interpolate_pos_encoding(self, x: jnp.ndarray, w: int, h: int, pos_embed: jnp.ndarray):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim),
            (1, w0, h0, dim),
            method="bicubic",
        )
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, -1, dim))

        return jnp.concatenate((class_pos_embed[None], patch_pos_embed), axis=1).astype(previous_dtype)

    def _get_rope_params(self, H, W, base, D_head, dtype):
        periods = base ** (
            2 * jnp.arange(D_head // 4, dtype=dtype) / (D_head // 2)
            )  # [D//4]

        coords_h = jnp.arange(0.5, H, dtype=dtype) / H  # [H]
        coords_w = jnp.arange(0.5, W, dtype=dtype) / W  # [W]

        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)  # [H, W, 2]
        coords = coords.reshape(-1, 2)
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.reshape(-1, D_head // 2)
        angles = jnp.tile(angles, 2)  # [HW, D]
        cos = jnp.cos(angles)  # [HW, D]
        sin = jnp.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def __call__(self, x, training: bool = False):
        B, H, W, C = x.shape
        assert H == W == self.img_size, "x size must be (B, {}, {}, {})".format(
            self.img_size, self.img_size, C
        )

        x = self.patch_embed(x)
        cls_token = jnp.broadcast_to(self.cls_token.value, (x.shape[0], *self.cls_token.value.shape[1:]))
        x = jnp.concatenate((cls_token, x), axis=1)

        if self.num_storage_tokens > 0:
            storage_token = jnp.broadcast_to(self.storage_tokens.value, (x.shape[0], *self.storage_token.value.shape[1:]))
            x = jnp.concatenate((x[:, :1], storage_token, x[:, 1:]), axis=1)

        w0 = W // self.patch_size
        h0 = H // self.patch_size
            
        rope = self._get_rope_params(h0, w0, 100, self.embed_dim // self.num_heads,
                                     x.dtype)
            
        for block in self.blocks:
            x = block(x, rope=rope, training=training)

        x_norm = self.norm(x)

        if self.num_storage_tokens > 0:
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1:self.num_storage_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_storage_tokens + 1:],
                "x_prenorm": x,
            }
        else:
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_patchtokens": x_norm[:, 1:],
                "x_prenorm": x,
            }

if __name__ == "__main__":

    num_heads = 6
    embed_dim = 384
    mlp_ratio = 4

    rngs = nnx.Rngs(params=0, dropout=1)

    vit_cls = functools.partial(
        DinoViT,
        num_heads=num_heads,
        embed_dim=embed_dim,
        mlp_ratio=mlp_ratio,
        depth=12,
        img_size=518,
        rngs=rngs,
    )
    vit_def = vit_cls()
    vit_params = nnx.state(vit_def)

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    params = load_vit_params(vit_params, dinov2_vits14)

    image = jax.random.uniform(jax.random.PRNGKey(0), (1, 518, 518, 3))
    nnx.update(vit_def, params)
    embed_jax = vit_def(image, training=False)
    embed_jax = onp.asarray(embed_jax["x_norm_patchtokens"])

    # Torch: forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_torch = torch.from_numpy(onp.asarray(image.transpose((0, 3, 1, 2))).copy()).to(device)
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    dinov2_vits14.eval()
    embed_torch = (
        dinov2_vits14.forward_features(image_torch)["x_norm_patchtokens"]
        .detach()
        .cpu()
        .numpy()
    )
    embed_torch2 = (
        dinov2_vits14.forward_features(torch.rand((1, 3, 518, 518), device=device))[
            "x_norm_patchtokens"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    cosine_distance = (
        onp.sum(embed_torch * embed_jax)
        / onp.linalg.norm(embed_torch)
        / onp.linalg.norm(embed_jax)
    )
    cosine_distance2 = (
        onp.sum(embed_torch2 * embed_jax)
        / onp.linalg.norm(embed_torch2)
        / onp.linalg.norm(embed_jax)
    )

    # Cosine distance for the first pair (same image) should be close to 1
    assert cosine_distance > 0.999, cosine_distance
    # Cosine distance for the second pair (different images) should be further away.
    # It might still be close to 1, because random noise is semantically similar.
    assert cosine_distance2 < 0.95, cosine_distance2
