# Response to Reviewer kJBN


## Full results on UEA datasets (29 datasets selected by NuTime). Our model outperforms both alternative methods. We will include the comparison in our final version.
|  | Ours (UNI.) | NuTime (self-supervised) | ITIME+SAN |
|---|:---:|:---:|:---:|
| ArticularyWordRecognition | 99.3 | 99.4 | 98.2 |
| AtrialFibrillation | 56.7 | 34.7 | 38.3 |
| BasicMotions | 100.0 | 100.0 | 100.0 |
| CharacterTrajectories | 99.7 | 99.4 | 99.5 |
| Cricket | 100.0 | 100.0 | 98.6 |
| DuckDuckGeese | 65.0 | 55.2 | 63.0 |
| EigenWorms | 85.5 | 91.0 | 83.2 |
| ERing | 91.9 | 98.6 | 89.6 |
| Epilepsy | 98.9 | 99.3 | 97.3 |
| EthanolConcentration | 39.2 | 46.6 | 37.4 |
| FaceDetection | 68.4 | 66.3 | 66.7 |
| FingerMovements | 65.0 | 61.2 | 59.5 |
| HandMovementDirection | 49.3 | 53.2 | 42.9 |
| Handwriting | 61.6 | 22.8 | 59.9 |
| Heartbeat | 81.0 | 78.4 | 79.9 |
| JapaneseVowels | 99.1 | 98.3 | 98.0 |
| Libras | 79.4 | 97.6 | 67.6 |
| LSST | 65.3 | 69.3 | 50.2 |
| MotorImagery | 65.0 | 62.2 | 60.7 |
| NATOPS | 98.9 | 94.0 | 96.8 |
| PEMS-SF | 79.2 | 92.5 | 64.6 |
| PenDigits | 97.6 | 98.8 | 95.7 |
| PhonemeSpectra | 31.3 | 32.0 | 30.5 |
| RacketSports | 89.8 | 93.4 | 87.5 |
| SelfRegulationSCP1 | 90.1 | 89.9 | 89.0 |
| SelfRegulationSCP2 | 59.4 | 60.3 | 58.5 |
| StandWalkJump | 63.3 | 66.7 | 56.7 |
| SpokenArabicDigits | 100.0 | 99.3 | 99.8 |
| UWaveGestureLibrary | 90.2 | 95.5 | 87.3 |
| Avg. | **78.3** | 77.8 | 74.4 |

# Response to Reviewer QtfR

## Single-backward vs Double-backward FIC
|Metrics|Accuracy(%)||Runtime(s)||
|:-:|:-:|:-:|:-:|:-:|
|Dataset|Double|Ours|Double|Ours|
|EC|39.0|39.2|0.096|0.046|
|FD|68.5|68.4|0.054|0.036|
|HW|64.0|61.6|0.432|0.182|
|HB|81.3|81|0.047|0.032|
|JV|99.2|99.1|0.041|0.028|
|SCP1|89.8|90.1|0.058|0.032|
|SCP2|59.7|59.4|0.067|0.036|
|SAD|100.0|100|0.185|0.091|
|UW|91.7|90.2|0.046|0.028|
|PS|81.3|79.2|0.445|0.185|
|**Avg.**|77.4|76.8|0.147|0.070|

# Response to Reviewer YXQa

## Updated Figure 1 with Caption

<img src="https://github.com/AnonymousUserss/ICML2025-4119-Response/blob/main/new_concept_fig.jpg" width="500" />





## Full results of explicit domain-shift datasets
|Dataset|Method|Acc.|Bal.Acc.|F1|P|R|
|---|---|---|---|---|---|---|
|TDBrain|Baseline|93.0|93.0|93.0|93.4|93.0|
||**+FIC**|**96.2**|**96.2**|**96.2**|**96.3**|**96.2**|
|ADFTD|Baseline|46.0|43.9|44.0|44.1|43.9|
||**+FIC**|**52.8**|**49.9**|**48.3**|**52.3**|**49.9**|
|PTB-XL|Baseline|73.9|59.2|60.0|65.4|59.2|
||**+FIC**|**75.1**|**61.4**|**63.0**|**68.6**|**61.4**|
|SleepEDF|Baseline|85.1|74.3|74.8|76.5|74.3|
||**+FIC**|**86.7**|**75.5**|**75.5**|**78.3**|**75.5**|
|OnHW-Char (R→L)|Baseline|44.1|44.1|45.2|51.3|44.1|
||**+FIC**|**46.8**|**46.8**|**47.6**|**53.5**|**46.8**|


Dataset sources:


| Dataset    | Source                                                                                                                                                                                                                                                                                                                                       |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TDBrain    | "The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database."                                                                                                                                                                           |
| ADFTD      | "A dataset of scalp EEG recordings of Alzheimer’s disease, frontotemporal dementia and healthy subjects from routine EEG."  "DICE-net: a novel convolution-transformer architecture for Alzheimer detection in EEG signals."  |
| PTB-XL     | "PTB-XL, a large publicly available electrocardiography dataset."                                                                                                                                                                                                                  |
| SleepEDF   |  "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG."                                                                                                         |
| OnHW-Chars | https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html. "The onhw dataset: Online handwriting recognition from imu-enhanced ballpoint pens with machine learning." "Benchmarking online sequence-to-sequence and character-based handwriting recognition from IMU-enhanced pens."                                                                                                                                                                                                                              |






