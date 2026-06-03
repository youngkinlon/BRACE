import torch


def force_scheduler(cache_dic, current):
    if cache_dic["fresh_ratio"] == 0:
        # FORA
        linear_step_weight = 0.0
    else:
        # TokenCache
        linear_step_weight = 0.0
    step_factor = torch.tensor(
        1 - linear_step_weight + 2 * linear_step_weight * current["step"] / current["num_steps"]
    )
    threshold = torch.round(cache_dic["fresh_threshold"] / step_factor)

    if torch.is_tensor(threshold):
        threshold_value = int(threshold.item())
    else:
        threshold_value = int(threshold)
    threshold_value = max(threshold_value, 1)

    # no force constrain for sensitive steps, cause the performance is good enough.
    # you may have a try.

    cache_dic["cal_threshold"] = threshold_value
    # return threshold
