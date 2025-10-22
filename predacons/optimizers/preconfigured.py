def get_adafactor(model_params, learning_rate, use_t5=False):
    from transformers import Adafactor

    if use_t5:
        optimizer = Adafactor(
            model_params, relative_step=True, warmup_init=True, lr=None
        )
    else:
        optimizer = Adafactor(
            model_params,
            lr=learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    return optimizer


def get_adamw(model_params, learning_rate):
    from transformers import AdamW

    return AdamW(model_params, lr=learning_rate)


def get_deepspeed_adamw(model_params, learning_rate):
    """Deep Speed's Implementation of ADAMW (CPU)
    DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
    In order to apply this optimizer, the model requires to have its master parameter
    (in FP32) reside on the CPU memory.
    See https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu for details
    """
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    return DeepSpeedCPUAdam(
        model_params=model_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        adamw_mode=True,
    )


def get_deepspeed_adam(model_params, learning_rate):
    """Deep Speed's Implementation of ADAM (CPU)
    DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
    In order to apply this optimizer, the model requires to have its master parameter
    (in FP32) reside on the CPU memory.
    See https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu for details
    """
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    return DeepSpeedCPUAdam(
        model_params=model_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        adamw_mode=False,
    )


def get_1b_adam(model_params, learning_rate):
    """(GPU)"""
    from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

    return OnebitAdam(params=model_params, lr=learning_rate)


def get_1b_lamb(model_params, learning_rate):
    from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

    return OnebitLamb(params=model_params, lr=learning_rate)


def get_fused_adam(model_params, learning_rate):
    from deepspeed.ops.adam import FusedAdam

    return FusedAdam(params=model_params, lr=learning_rate)


def get_fused_lamb(model_params, learning_rate):
    from deepspeed.ops.lamb import FusedLamb

    return FusedLamb(params=model_params, lr=learning_rate)