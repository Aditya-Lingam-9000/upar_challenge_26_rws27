# UPAR Challenge — Training Results Summary

The training log contains hardware details, dataset statistics, model configuration, baseline evaluation, epoch-level training results, validation metrics, and checkpoint information. The repeated distributed-process logs and individual step-by-step epoch logs are excluded.

## 1. Hardware and environment

| Parameter                       | Result                 |
| ------------------------------- | ---------------------- |
| Platform                        | Kaggle Notebook        |
| GPUs                            | 2 × NVIDIA Tesla T4   |
| VRAM per GPU                    | 14.56 GB               |
| Total GPU VRAM                  | 29.12 GB               |
| CPU cores                       | 4                      |
| System RAM                      | 31.35 GB               |
| Available RAM at initialization | 29.83 GB               |
| CUDA available                  | Yes                    |
| Mixed precision                 | FP16 during evaluation |
| Distributed training            | PyTorch DDP, 2 GPUs    |
| Model backbone                  | OpenCLIP EVA02-B-16    |
| Pretrained weights              | `merged2b_s8b_b131k` |

## 2. Dataset statistics

| Parameter            | Training set               | Validation set             |
| -------------------- | -------------------------- | -------------------------- |
| Number of samples    | 97,669                     | 33,407                     |
| RAM cache size       | 473.0 MB                   | 161.1 MB                   |
| Dataset format       | HDF5                       | HDF5                       |
| Dataset file         | `upar_binary_dataset.h5` | `upar_binary_dataset.h5` |
| Cache initialization | 14.56 seconds              | 4.77 seconds               |
| Validation queries   | —                         | 500                        |

## 3. Training configuration

| Parameter                    | Configured value                        |
| ---------------------------- | --------------------------------------- |
| Planned epochs               | 25                                      |
| Batch size per GPU           | 96                                      |
| Global batch size            | 192                                     |
| Effective batch size         | 384                                     |
| Gradient accumulation steps  | 2                                       |
| Backbone learning rate       | 0.00005                                 |
| Adapter learning rate        | 0.00035                                 |
| Minimum learning rate        | 0.000001                                |
| Evaluation frequency         | Every epoch                             |
| Evaluation batch size        | 384                                     |
| Maximum session budget       | 10.75 hours                             |
| Training steps per epoch     | 508                                     |
| Total planned training steps | 12,700                                  |
| Checkpoint directory         | `checkpoints_sota`                    |
| Best-model checkpoint        | `checkpoints_sota/best_model.pt`      |
| Latest checkpoint            | `checkpoints_sota/last_checkpoint.pt` |

## 4. Baseline evaluation — before fine-tuning

| Evaluation pathway         | mADM   | Rank-1 | Rank-5 | mAP   | mINP  |
| -------------------------- | ------ | ------ | ------ | ----- | ----- |
| Continuous Metric (1024-D) | 0.1297 | 0.40%  | 1.40%  | 0.86% | —    |
| Attribute Surprisal L1     | 0.1523 | 2.40%  | —     | 1.11% | —    |
| Dual-Pathway RRF Fusion    | 0.1353 | 1.20%  | 2.80%  | 0.96% | 0.77% |

## 5. Epoch-wise training and validation results

The table contains one row per completed epoch. Each epoch's step-by-step logs are omitted. The validation values are from the competition evaluation summaries in the supplied log.

| Epoch | Mean training loss | Continuous mADM | Cont. R-1 | Cont. R-5 | Cont. mAP | Attribute mADM | Attr. R-1 | Attr. mAP | Fusion mADM | Fusion R-1 | Fusion R-5 | Fusion mAP | Fusion mINP |
| ----- | ------------------ | --------------- | --------- | --------- | --------- | -------------- | --------- | --------- | ----------- | ---------- | ---------- | ---------- | ----------- |
| 1     | 6.1463             | 0.6053          | 38.20%    | 75.20%    | 29.23%    | 0.3091         | 21.40%    | 6.76%     | 0.5806      | 50.60%     | 75.60%     | 27.26%     | 3.16%       |
| 2     | 3.5571             | 0.6668          | 54.80%    | 84.60%    | 37.77%    | 0.3350         | 29.40%    | 8.08%     | 0.6415      | 62.20%     | 84.20%     | 34.37%     | 4.03%       |
| 3     | 3.1255             | 0.6909          | 59.20%    | 87.20%    | 41.61%    | 0.3299         | 30.40%    | 8.65%     | 0.6607      | 68.40%     | 87.40%     | 37.90%     | 4.71%       |
| 4     | 2.8959             | 0.6737          | 60.00%    | 87.60%    | 39.28%    | 0.3415         | 38.20%    | 10.82%    | 0.6525      | 67.60%     | 86.60%     | 36.36%     | 4.21%       |
| 5     | 2.6970             | 0.6877          | 62.20%    | 89.00%    | 41.36%    | 0.3631         | 38.40%    | 12.09%    | 0.6646      | 67.20%     | 86.60%     | 37.87%     | 4.52%       |
| 6     | 2.5510             | 0.6781          | 57.60%    | 87.00%    | 39.62%    | 0.3866         | 44.20%    | 14.01%    | 0.6618      | 64.60%     | 85.60%     | 37.04%     | 3.92%       |
| 7     | 2.4217             | 0.6660          | 54.60%    | 86.80%    | 38.26%    | 0.3965         | 44.20%    | 14.61%    | 0.6539      | 63.20%     | 85.00%     | 36.19%     | 3.92%       |
| 8     | 2.2911             | 0.6784          | 58.20%    | 87.80%    | 40.19%    | 0.4268         | 48.80%    | 16.87%    | 0.6672      | 64.60%     | 87.20%     | 37.95%     | 4.04%       |
| 9     | 2.1668             | 0.6733          | 57.60%    | 87.00%    | 39.22%    | 0.4267         | 47.00%    | 16.59%    | 0.6599      | 64.40%     | 87.20%     | 37.02%     | 4.20%       |
| 10    | 2.0824             | 0.6738          | 56.60%    | 86.80%    | 39.35%    | 0.4483         | 59.40%    | 18.24%    | 0.6624      | 66.40%     | 87.40%     | 37.39%     | 4.02%       |
| 11    | 2.0071             | 0.6807          | 59.00%    | 89.60%    | 40.32%    | 0.4773         | 54.60%    | 19.88%    | 0.6709      | 65.00%     | 88.20%     | 38.48%     | 4.25%       |
| 12    | 1.9143             | 0.6749          | 58.00%    | 91.40%    | 39.59%    | 0.4859         | 47.80%    | 21.09%    | 0.6684      | 65.20%     | 88.00%     | 38.34%     | 4.21%       |
| 13    | 1.8472             | 0.6731          | 55.60%    | 90.20%    | 39.32%    | 0.5118         | 57.40%    | 22.82%    | 0.6696      | 63.40%     | 90.00%     | 38.72%     | 4.56%       |
| 14    | 1.7998             | 0.6760          | 56.20%    | 89.80%    | 39.84%    | 0.5211         | 60.40%    | 23.84%    | 0.6718      | 67.80%     | 90.20%     | 39.10%     | 4.54%       |
| 15    | 1.7421             | 0.6770          | 58.00%    | 89.40%    | 39.76%    | 0.5364         | 63.20%    | 24.82%    | 0.6742      | 65.40%     | 89.40%     | 39.18%     | 4.44%       |
| 16    | 1.6941             | 0.6737          | 57.20%    | 90.20%    | 39.61%    | 0.5431         | 63.60%    | 25.75%    | 0.6723      | 61.20%     | 90.80%     | 39.16%     | 4.46%       |
| 17    | 1.6594             | 0.6720          | 57.40%    | 88.80%    | 39.24%    | 0.5498         | 62.80%    | 26.32%    | 0.6710      | 65.40%     | 90.80%     | 38.86%     | 4.73%       |
| 18    | 1.6408             | 0.6724          | 56.00%    | 89.00%    | 39.27%    | 0.5567         | 66.00%    | 27.18%    | 0.6716      | 63.60%     | 90.00%     | 39.01%     | 4.48%       |
| 19    | 1.6155             | 0.6747          | 60.80%    | 89.00%    | 39.66%    | 0.5603         | 62.00%    | 27.34%    | 0.6745      | 67.20%     | 90.20%     | 39.60%     | 4.57%       |
| 20    | 1.5926             | 0.6748          | 55.00%    | 90.80%    | 39.29%    | 0.5653         | 62.40%    | 27.71%    | 0.6751      | 65.20%     | 91.40%     | 39.43%     | 4.67%       |
| 21    | 1.5709             | 0.6765          | 57.40%    | 88.80%    | 39.62%    | 0.5712         | 62.20%    | 28.14%    | 0.6760      | 64.40%     | 91.20%     | 39.59%     | 4.68%       |
| 22    | 1.5571             | 0.6766          | 58.00%    | 89.40%    | 39.80%    | 0.5731         | 65.00%    | 28.51%    | 0.6773      | 63.60%     | 92.40%     | 39.94%     | 4.75%       |
| 23    | 1.5593             | 0.6780          | 59.00%    | 88.40%    | 40.07%    | 0.5736         | 63.60%    | 28.53%    | 0.6769      | 60.60%     | 91.60%     | 39.87%     | 4.74%       |
| 24    | 1.5428             | 0.6776          | 60.00%    | 87.00%    | 40.05%    | 0.5745         | 65.80%    | 28.64%    | 0.6765      | 62.60%     | 90.60%     | 39.81%     | 4.70%       |
| 25    | 1.5486             | 0.6775          | 60.40%    | 88.40%    | 40.08%    | 0.5747         | 64.40%    | 28.69%    | 0.6770      | 63.80%     | 90.80%     | 39.89%     | 4.73%       |

R-1 = Rank-1 retrieval accuracy; R-5 = Rank-5 retrieval accuracy; mAP = mean Average Precision; mINP = mean Inverse Negative Penalty. mADM values are reproduced as reported in the log.

## 6. Best validation performance

| Metric                    | Best reported result | Epoch |
| ------------------------- | -------------------- | ----- |
| Fusion mADM               | 0.6773               | 22    |
| Fusion Rank-1             | 68.40%               | 3     |
| Fusion Rank-5             | 92.40%               | 22    |
| Fusion mAP                | 39.94%               | 22    |
| Fusion mINP               | 4.75%                | 22    |
| Continuous mADM           | 0.6909               | 3     |
| Continuous Rank-1         | 62.20%               | 5     |
| Continuous Rank-5         | 91.40%               | 12    |
| Continuous mAP            | 41.61%               | 3     |
| Attribute Rank-1          | 66.00%               | 18    |
| Attribute mAP             | 28.69%               | 25    |
| Lowest mean training loss | 1.5428               | 24    |

These are the best values across the reported epochs, not necessarily values achieved by one single checkpoint.

## 7. Checkpoint and model integrity audit

| Check                          | Result                      |
| ------------------------------ | --------------------------- |
| Best model checkpoint          | `best_model.pt`           |
| Latest checkpoint              | `last_checkpoint.pt`      |
| Best checkpoint epoch          | 22                          |
| Best checkpoint step           | 11,176                      |
| Latest checkpoint epoch        | 25                          |
| Latest checkpoint step         | 12,700                      |
| Best checkpoint size           | 2,820.60 MB                 |
| Latest checkpoint size         | 2,820.61 MB                 |
| Multi-session state components | All 7 verified              |
| Model loading test             | Passed for both checkpoints |
| Forward-pass verification      | Passed                      |
| Intermediate checkpoint bloat  | None reported               |
| Final audit                    | Passed                      |

## 8. Submission packaging and compliance

| Parameter                | Result                         |
| ------------------------ | ------------------------------ |
| Submission archive       | `submission.zip`             |
| Archive size             | 434.73 MB                      |
| FP16 model weights       | 470.16 MB                      |
| Extracted FP16 tensors   | 473                            |
| Visual tensors           | 280                            |
| Text tensors             | 155                            |
| NaN/Inf weights          | 0 reported                     |
| `run.py`               | Passed                         |
| `metadata.yaml`        | Passed                         |
| Attribute priors         | Passed; shape `(40,)`        |
| Vendored `timm` module | Passed                         |
| CodaBench entry points   | Both confirmed                 |
| Air-gapped code hygiene  | Passed                         |
| Archive size limit       | Passed; below 500 MB           |
| Structural audit         | 10/10 checks passed            |
| Packaging status         | Ready for CodaBench evaluation |

## 9. Overall execution summary

| Parameter                     | Final result                      |
| ----------------------------- | --------------------------------- |
| Planned epochs                | 25                                |
| Completed epochs              | 25                                |
| Completed training steps      | 12,700                            |
| Total logged elapsed time     | Approximately 8 hours, 22 minutes |
| Final mean training loss      | 1.5486                            |
| Historical best fusion mADM   | 0.6773                            |
| Best checkpoint epoch         | 22                                |
| Latest checkpoint             | Successfully saved                |
| Submission archive            | Successfully generated            |
| Post-training integrity audit | Passed                            |
| Isolated submission audit     | Passed, 10/10                     |
| External competition result   | Not included in the supplied log  |

Final status: Training completed all 25 epochs. The best validation checkpoint was saved at epoch 22, and the submission archive passed the reported structural checks. The log establishes package readiness, not an official leaderboard placement or independently confirmed competition score.
