sample = 1
import numpy as np


def autoregressive_sampling(x, model, N):
    n = len(x)
    T = len(x) + N

    while n < T:
        x = np.append(x, sample(model(x)[-1]))
        n += 1

    return x


def max_fn(x):
    # Set negative values to zero and normalize
    x_max = np.where(x > 0, x, 0)
    return x_max / np.sum(x_max)


def speculative_sampling(x, draft_model, target_model, N, K):
    n = len(x)
    T = len(x) + N

    while n < T:
        # Step 1: Generate K tokens from the draft model using autoregressive decoding
        x_draft = x.copy()  # Make a copy of x
        for _ in range(K):
            p = draft_model(x_draft)
            x_draft = np.append(x_draft, sample(p[-1]))  # Append a sampled token

        # Step 2: Get the predictions from the target model for the draft sequence
        q = target_model(x_draft)

        # Step 3: Decide whether to accept or reject the draft tokens
        all_accepted = True
        for _ in range(K):
            i = n - 1
            j = x_draft[i + 1]
            acceptance_prob = min(1, q[i][j] / p[i][j])
            if np.random.random() < acceptance_prob:  # Accept the token
                x = np.append(x, j)
                n += 1
            else:  # Reject the token and resample
                resampled_token = sample(max_fn(q[i] - p[i]))
                x = np.append(x, resampled_token)
                n += 1
                all_accepted = False
                break

        # Step 4: If all draft tokens were accepted, sample an additional final token
        if all_accepted:
            final_token = sample(q[-1])
            x = np.append(x, final_token)
            n += 1

        # Sanity check
        assert n == len(x), f"Token count mismatch: {n} {len(x)}"

    return x
