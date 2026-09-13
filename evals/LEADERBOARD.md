# Leaderboard

Generated 2026-09-13 15:22 from `evals/runs.db`. One row per finished run+condition; all metrics.


## merchant_embed_vocab

| run             | model            | commit  | condition        | bank_heldout | bank_train | desc_heldout | desc_train | name_heldout | name_train |
|-----------------|------------------|---------|------------------|--------------|------------|--------------|------------|--------------|------------|
| 20260913-151609 | all-MiniLM-L6-v2 | eb845a6 | ft_newtok_mean   | 4.2          | 26.0       | 100.0        | 100.0      | 12.5         | 100.0      |
| 20260913-151609 | all-MiniLM-L6-v2 | eb845a6 | ft_newtok_random | 8.3          | 22.9       | 100.0        | 100.0      | 8.3          | 100.0      |
| 20260913-151609 | all-MiniLM-L6-v2 | eb845a6 | ft_subword       | 4.2          | 70.8       | 100.0        | 100.0      | 12.5         | 100.0      |
| 20260913-151609 | all-MiniLM-L6-v2 | eb845a6 | zero_shot        | 4.2          | 7.3        | 95.8         | 99.0       | 29.2         | 6.2        |

Configs:
- `20260913-151609`: `{"bs": 32, "epochs": 6, "lr": 3e-05, "n_heldout": 24, "n_merchants": 120, "seed": 0}`

## merchant_knowledge_injection

| run             | model        | commit  | condition              | bank_category | bank_category_upper_alias | clean_category | minutes | ppl_general | reverse | sells |
|-----------------|--------------|---------|------------------------|---------------|---------------------------|----------------|---------|-------------|---------|-------|
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | base                   | 8.3           |                           | 8.3            | 0.6     | 16.15       | 31.7    | 20.0  |
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | ft_aug                 | 8.3           |                           | 9.2            | 2.5     | 795.56      | 46.7    | 46.7  |
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | ft_aug_lora            | 12.5          |                           | 17.5           | 2.7     | 201.96      | 51.7    | 60.8  |
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | ft_aug_vocab           | 8.3           | 8.3                       | 10.0           | 2.0     | 885.82      | 79.2    | 31.7  |
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | ft_aug_wise0.5         | 8.3           |                           | 25.0           |         | 28.08       | 47.5    | 56.7  |
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | ft_raw                 | 10.0          |                           | 15.0           | 1.8     | 420.89      | 31.7    | 45.8  |
| 20260913-151657 | Qwen2.5-0.5B | 05e88a2 | incontext              | 14.2          |                           | 69.2           | 0.6     | 16.15       | 100.0   | 100.0 |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_aug_lr1e-05         | 12.5          |                           | 41.7           | 2.4     | 26.03       | 42.5    | 51.7  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_aug_lr1e-05_wise0.5 | 9.2           |                           | 26.7           |         | 17.18       | 30.8    | 52.5  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_aug_lr2e-05         | 13.3          |                           | 20.8           | 2.4     | 51.4        | 55.8    | 50.8  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_aug_lr2e-05_wise0.5 | 8.3           |                           | 40.8           |         | 19.11       | 38.3    | 68.3  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_raw_lr1e-05         | 8.3           |                           | 21.7           | 2.4     | 19.21       | 25.8    | 37.5  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_raw_lr1e-05_wise0.5 | 8.3           |                           | 9.2            |         | 16.48       | 27.5    | 26.7  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_raw_lr2e-05         | 8.3           |                           | 36.7           | 2.4     | 27.54       | 30.0    | 45.0  |
| 20260913-151719 | Qwen2.5-0.5B | 6dc7f71 | ft_raw_lr2e-05_wise0.5 | 8.3           |                           | 17.5           |         | 17.75       | 25.8    | 37.5  |

Configs:
- `20260913-151657`: `{"bs": 16, "lora_r": 64, "lr_full": 5e-05, "lr_lora": 0.0003, "n_merchants": 120, "seed": 0, "steps": 420}`
- `20260913-151719`: `{"bs": 16, "lr_full": [1e-05, 2e-05], "n_merchants": 120, "seed": 0, "steps": 420}`

## universe_embed

| run             | model            | commit   | condition | proto_habitat_k1_heldout | proto_habitat_k1_seen | proto_habitat_k3_seen | proto_type_k1_heldout | proto_type_k1_seen | proto_type_k3_seen | proto_weakness_k1_heldout | proto_weakness_k1_seen | proto_weakness_k3_seen | type_canonical_heldout | type_canonical_seen | type_synonym2_heldout | type_synonym2_seen | type_synonym_heldout | type_synonym_seen |
|-----------------|------------------|----------|-----------|--------------------------|-----------------------|-----------------------|-----------------------|--------------------|--------------------|---------------------------|------------------------|------------------------|------------------------|---------------------|-----------------------|--------------------|----------------------|-------------------|
| 20260913-151637 | all-MiniLM-L6-v2 | 0ca04ae* | trained   | 34.3                     | 45.0                  | 63.0                  | 36.3                  | 69.0               | 85.3               | 34.0                      | 79.0                   | 85.0                   | 8.3                    | 99.3                | 12.5                  | 24.3               | 20.8                 | 41.2              |
| 20260913-151637 | all-MiniLM-L6-v2 | 0ca04ae* | zero_shot | 34.0                     | 32.7                  | 25.7                  | 39.3                  | 38.0               | 33.7               | 48.0                      | 38.0                   | 33.3                   | 25.0                   | 10.3                | 8.3                   | 14.0               | 25.0                 | 13.2              |

Configs:
- `20260913-151637`: `{"bs": 32, "epochs": 8, "lr": 3e-05, "n_heldout": 24, "n_species": 160, "objective": "infonce_name_to_attribute_text", "seed": 0}`
