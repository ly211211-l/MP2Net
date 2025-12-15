**1.检查已保存的模型：**

cd /mnt/d/github/MP2Net\_exp

ls -lth ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/\*.pth





-rwxrwxrwx 1 ly ly 89M Dec  3 06:39 ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/model\_2.pth

-rwxrwxrwx 1 ly ly 89M Dec  3 06:39 ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/model\_last.pth

-rwxrwxrwx 1 ly ly 89M Dec  2 20:20 ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/model\_1.pth







**2.记录训练状态**

\# 查看训练日志，记录当前epoch

tail -20 ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/log.txt



\# 查看训练配置

cat ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/opt.txt





(base) ly@LAPTOP-JPT06F1I:/mnt/d/github/MP2Net\_exp$ # 查看训练日志，记录当前epoch

tail -20 ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/log.txt



\# 查看训练配置

cat ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/opt.txt

2025-12-02-20-20: epoch: 1 |loss 5.646465 | hm\_loss 2.343528 | wh\_loss 1.702252 | off\_loss 0.236717 | track\_loss 0.491740 | seq\_loss 2.404255 | time 0.016667 |

2025-12-03-06-39: epoch: 2 |loss 4.437072 | hm\_loss 1.758706 | wh\_loss 1.516937 | off\_loss 0.225482 | track\_loss 0.380105 | seq\_loss 1.921085 | time 0.016667 |

==> torch version: 2.4.1+cu121

==> cudnn version: 90100

==> Cmd:

\['train.py', '--model\_name', 'DLADCN', '--gpus', '0', '--lr', '1.25e-4', '--lr\_step', '14', '--num\_epochs', '15', '--batch\_size', '1', '--seqLen', '5', '--datasetname', 'ICPR', '--data\_dir', './dataset/ICPR/', '--num\_workers', '3']

==> Opt:

&nbsp; K: 450

&nbsp; batch\_size: 1

&nbsp; conf\_thres: 0.3

&nbsp; dataName: ICPR

&nbsp; data\_dir: ./dataset/ICPR/

&nbsp; datasetname: ICPR

&nbsp; device: cuda

&nbsp; down\_ratio: 1

&nbsp; gpus: \[0]

&nbsp; gpus\_str: 0

&nbsp; load\_model:

&nbsp; lr: 0.000125

&nbsp; lr\_step: \[14]

&nbsp; model\_name: DLADCN

&nbsp; nms: False

&nbsp; nms\_thres: 0.4

&nbsp; num\_classes: 1

&nbsp; num\_epochs: 15

&nbsp; num\_workers: 3

&nbsp; resume: False

&nbsp; save\_dir: ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19

&nbsp; save\_log\_dir: ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19

&nbsp; save\_results\_dir: ./ICPR/DLADCN/results

&nbsp; save\_track\_results: True

&nbsp; seed: 317

&nbsp; seqLen: 5

&nbsp; show\_results: False

&nbsp; val\_intervals: 100





**3.后续恢复训练**

**python train.py \\**

    **--model\_name DLADCN \\**

    **--gpus 0 \\**

    **--load\_model ./ICPR/DLADCN/weights2025\_12\_02\_10\_30\_19/model\_last.pth \\**

    **--resume True \\**

    **--lr 1.25e-4 \\**

    **--lr\_step 14 \\**

    **--num\_epochs 15 \\**

    **--batch\_size 1 \\**

    **--seqLen 5 \\**

    **--datasetname ICPR \\**

    **--data\_dir ./dataset/ICPR/ \\**

    **--num\_workers 3**

