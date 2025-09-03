1. 3 q 最好的moe

python train_single.py\
 --exp_name 3q_128_cls_moe_16h\
 --num_qubits 3\
 --d_model 128\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 16\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4

表现很差：

📊 [3q_128_cls_moe_16h] Epoch  4|val   | Loss: 0.018033 | MAE: 0.119183 | R²: -0.0246          
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.018033
💪 [3q_128_cls_moe_16h] Epoch  5|train | Loss: 0.018897                                        
📊 [3q_128_cls_moe_16h] Epoch  5|val   | Loss: 0.018005 | MAE: 0.119040 | R²: -0.0230          
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.018005
💪 [3q_128_cls_moe_16h] Epoch  6|train | Loss: 0.019035                                        
📊 [3q_128_cls_moe_16h] Epoch  6|val   | Loss: 0.017918 | MAE: 0.118570 | R²: -0.0181          
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.017918
💪 [3q_128_cls_moe_16h] Epoch  7|train | Loss: 0.018909                                        
📊 [3q_128_cls_moe_16h] Epoch  7|val   | Loss: 0.017674 | MAE: 0.116813 | R²: -0.0043          
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.017674
💪 [3q_128_cls_moe_16h] Epoch  8|train | Loss: 0.018991                                        
📊 [3q_128_cls_moe_16h] Epoch  8|val   | Loss: 0.017639 | MAE: 0.116482 | R²: -0.0023                                                                    -04]
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.017639                                                                                         p-25
💪 [3q_128_cls_moe_16h] Epoch  9|train | Loss: 0.018949                                        
📊 [3q_128_cls_moe_16h] Epoch  9|val   | Loss: 0.018314 | MAE: 0.120489 | R²: -0.0404          
💪 [3q_128_cls_moe_16h] Epoch 10|train | Loss: 0.018810                                        
📊 [3q_128_cls_moe_16h] Epoch 10|val   | Loss: 0.018176 | MAE: 0.119887 | R²: -0.0326          
💪 [3q_128_cls_moe_16h] Epoch 11|train | Loss: 0.018837                                        
📊 [3q_128_cls_moe_16h] Epoch 11|val   | Loss: 0.017636 | MAE: 0.116413 | R²: -0.0021          
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.017636
💪 [3q_128_cls_moe_16h] Epoch 12|train | Loss: 0.018808                                        
📊 [3q_128_cls_moe_16h] Epoch 12|val   | Loss: 0.017770 | MAE: 0.117628 | R²: -0.0097          
💪 [3q_128_cls_moe_16h] Epoch 13|train | Loss: 0.018599                                        
📊 [3q_128_cls_moe_16h] Epoch 13|val   | Loss: 0.018265 | MAE: 0.120281 | R²: -0.0377          
💪 [3q_128_cls_moe_16h] Epoch 14|train | Loss: 0.018753                                        
📊 [3q_128_cls_moe_16h] Epoch 14|val   | Loss: 0.017877 | MAE: 0.118352 | R²: -0.0157          
💪 [3q_128_cls_moe_16h] Epoch 15|train | Loss: 0.018786                                        
📊 [3q_128_cls_moe_16h] Epoch 15|val   | Loss: 0.017613 | MAE: 0.116157 | R²: -0.0009          
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.017613
💪 [3q_128_cls_moe_16h] Epoch 16|train | Loss: 0.018610                                        
📊 [3q_128_cls_moe_16h] Epoch 16|val   | Loss: 0.017957 | MAE: 0.118803 | R²: -0.0202          
💪 [3q_128_cls_moe_16h] Epoch 17|train | Loss: 0.018630                                        
📊 [3q_128_cls_moe_16h] Epoch 17|val   | Loss: 0.017858 | MAE: 0.118249 | R²: -0.0147          
💪 [3q_128_cls_moe_16h] Epoch 18|train | Loss: 0.018480                                        
📊 [3q_128_cls_moe_16h] Epoch 18|val   | Loss: 0.017759 | MAE: 0.117594 | R²: -0.0090          
💪 [3q_128_cls_moe_16h] Epoch 19|train | Loss: 0.018590                                        
📊 [3q_128_cls_moe_16h] Epoch 19|val   | Loss: 0.017681 | MAE: 0.116977 | R²: -0.0046          
🚀 [3q_128_cls_moe_16h] Train Epoch 20:  50%|██████████████████████████▌                          | 8/16 [00:05<00:03,  2.29it/s, loss=0.018276, lr=1.00e🚀 [3q_128_cls_moe_16h] Train Epoch 20:  50%|████████████████████████▌                        | 8/16 [00:07<00:03,  2.29it/s, loss=0.018635, lr=1.00e-04]💪 [3q_128_cls_moe_16h] Epoch 20|train | Loss: 0.018622                                                                                                  
📊 [3q_128_cls_moe_16h] Epoch 20|val   | Loss: 0.017570 | MAE: 0.115093 | R²: 0.0015                                                                     
✅ [3q_128_cls_moe_16h] New best model saved! Val Loss: 0.017570
🔓 Cleaned up GPU lock: /tmp/gpu_locks/gpu_0_3q_128_cls_moe_16h.lock

🧪 [3q_128_cls_moe_16h] Final Test Evaluation...
/home/gwl/3q/MixMLP/train_single.py:350: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path_manager.model_path)) # Load best model
🔍 Evaluating: 100%|██████████| 2/2 [00:02<00:00,  1.12s/it]
📋 [3q_128_cls_moe_16h] Final Test Results:
   MSE: 0.017547
   MAE: 0.114780
   R²:  -0.0027
   Spearman: 0.7458

✅ Experiment 3q_128_cls_moe_16h completed successfully!
📁 All outputs saved to: experiments/3q_128_cls_moe_16h_20250902_175135
📄 Training log: experiments/3q_128_cls_moe_16h_20250902_175135/3q_128_cls_moe_16h.csv
📄 Final metrics: experiments/3q_128_cls_moe_16h_20250902_175135/3q_128_cls_moe_16h_final_metrics.json
🤖 Model weights: models/best_3q_128_cls_moe_16h_20250902_175135.pth
⚙️ Configuration: experiments/3q_128_cls_moe_16h_20250902_175135/3q_128_cls_moe_16h_config.json

+ epoch - 100
+ less head 4 and 8
+ D_model 128*2 256
+ mlp standard

2. num(head)??
python train_single.py\
 --exp_name 3q_128_cls_moe_64h\
 --num_qubits 3\
 --d_model 128\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 64\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4
看起来也不好，来点实际的。
💪 [3q_128_cls_moe_64h] Epoch  1|train | Loss: 0.030072                                         
📊 [3q_128_cls_moe_64h] Epoch  1|val   | Loss: 0.019533 | MAE: 0.124718 | R²: -0.1095           
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.019533
💪 [3q_128_cls_moe_64h] Epoch  2|train | Loss: 0.019742                                         
📊 [3q_128_cls_moe_64h] Epoch  2|val   | Loss: 0.018340 | MAE: 0.120592 | R²: -0.0419           
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.018340
💪 [3q_128_cls_moe_64h] Epoch  3|train | Loss: 0.019098                                         
📊 [3q_128_cls_moe_64h] Epoch  3|val   | Loss: 0.018024 | MAE: 0.119128 | R²: -0.0240           
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.018024
💪 [3q_128_cls_moe_64h] Epoch  4|train | Loss: 0.019081                                         
📊 [3q_128_cls_moe_64h] Epoch  4|val   | Loss: 0.017832 | MAE: 0.118024 | R²: -0.0132           
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.017832
💪 [3q_128_cls_moe_64h] Epoch  5|train | Loss: 0.018808                                         
📊 [3q_128_cls_moe_64h] Epoch  5|val   | Loss: 0.017913 | MAE: 0.118528 | R²: -0.0178           
💪 [3q_128_cls_moe_64h] Epoch  6|train | Loss: 0.018864                                         
📊 [3q_128_cls_moe_64h] Epoch  6|val   | Loss: 0.017604 | MAE: 0.115747 | R²: -0.0004           
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.017604
💪 [3q_128_cls_moe_64h] Epoch  7|train | Loss: 0.018804                                         
📊 [3q_128_cls_moe_64h] Epoch  7|val   | Loss: 0.017620 | MAE: 0.116100 | R²: -0.0013           
💪 [3q_128_cls_moe_64h] Epoch  8|train | Loss: 0.018873                                         
📊 [3q_128_cls_moe_64h] Epoch  8|val   | Loss: 0.017667 | MAE: 0.116743 | R²: -0.0039           
💪 [3q_128_cls_moe_64h] Epoch  9|train | Loss: 0.018831                                         
📊 [3q_128_cls_moe_64h] Epoch  9|val   | Loss: 0.017777 | MAE: 0.117672 | R²: -0.0101           
💪 [3q_128_cls_moe_64h] Epoch 10|train | Loss: 0.018624                                         
📊 [3q_128_cls_moe_64h] Epoch 10|val   | Loss: 0.017796 | MAE: 0.117809 | R²: -0.0111           
💪 [3q_128_cls_moe_64h] Epoch 11|train | Loss: 0.018596                                         
📊 [3q_128_cls_moe_64h] Epoch 11|val   | Loss: 0.017908 | MAE: 0.118521 | R²: -0.0175           
💪 [3q_128_cls_moe_64h] Epoch 12|train | Loss: 0.018593                                         
📊 [3q_128_cls_moe_64h] Epoch 12|val   | Loss: 0.017914 | MAE: 0.118559 | R²: -0.0178           
💪 [3q_128_cls_moe_64h] Epoch 13|train | Loss: 0.018576                                         
📊 [3q_128_cls_moe_64h] Epoch 13|val   | Loss: 0.017968 | MAE: 0.118865 | R²: -0.0209           
💪 [3q_128_cls_moe_64h] Epoch 14|train | Loss: 0.018580                                         
📊 [3q_128_cls_moe_64h] Epoch 14|val   | Loss: 0.017917 | MAE: 0.118586 | R²: -0.0180           
💪 [3q_128_cls_moe_64h] Epoch 15|train | Loss: 0.018588                                         
📊 [3q_128_cls_moe_64h] Epoch 15|val   | Loss: 0.017593 | MAE: 0.115844 | R²: 0.0003            
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.017593
💪 [3q_128_cls_moe_64h] Epoch 16|train | Loss: 0.018735                                         
📊 [3q_128_cls_moe_64h] Epoch 16|val   | Loss: 0.017903 | MAE: 0.118523 | R²: -0.0172           
💪 [3q_128_cls_moe_64h] Epoch 17|train | Loss: 0.018719                                         
📊 [3q_128_cls_moe_64h] Epoch 17|val   | Loss: 0.017624 | MAE: 0.116410 | R²: -0.0014           
💪 [3q_128_cls_moe_64h] Epoch 18|train | Loss: 0.018609                                         
📊 [3q_128_cls_moe_64h] Epoch 18|val   | Loss: 0.018108 | MAE: 0.119587 | R²: -0.0288           
💪 [3q_128_cls_moe_64h] Epoch 19|train | Loss: 0.018585                                         
📊 [3q_128_cls_moe_64h] Epoch 19|val   | Loss: 0.017566 | MAE: 0.115458 | R²: 0.0018            
✅ [3q_128_cls_moe_64h] New best model saved! Val Loss: 0.017566
💪 [3q_128_cls_moe_64h] Epoch 20|train | Loss: 0.018631                                         
📊 [3q_128_cls_moe_64h] Epoch 20|val   | Loss: 0.017794 | MAE: 0.117894 | R²: -0.0110 
？ 其实居然还行！！！！！咋办呢？、
那我们换成128？？

3. 史无前例的增大dmodel
python train_single.py\
 --exp_name 3q_256_cls_moe_64h\
 --num_qubits 3\
 --d_model 256\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 32\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4


4. 128-128; this is indeed crazy!!
python train_single.py\
 --exp_name 3q_128_cls_moe_128h\
 --num_qubits 3\
 --d_model 128\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 4\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4

this converges fast; we need to think on, rather naive tho, the ratio of dimention
"
✅ Experiment 3q_128_cls_moe_128h completed successfully!
📁 All outputs saved to: experiments/3q_128_cls_moe_128h_20250902_180934
📄 Training log: experiments/3q_128_cls_moe_128h_20250902_180934/3q_128_cls_moe_128h.csv
📄 Final metrics: experiments/3q_128_cls_moe_128h_20250902_180934/3q_128_cls_moe_128h_final_metrics.json
🤖 Model weights: models/best_3q_128_cls_moe_128h_20250902_180934.pth
⚙️ Configuration: experiments/3q_128_cls_moe_128h_20250902_180934/3q_128_cls_moe_128h_config.json

🔍 To visualize this model, run:
python Visualization.py \
    --model_path 'models/best_3q_128_cls_moe_128h_20250902_180934.pth' \
    --data_folder 'Decoded_Tokens' \
    --labels_path 'Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy' \
    --config_from_json 'experiments/3q_128_cls_moe_128h_20250902_180934/3q_128_cls_moe_128h_config.json' \
    --output_prefix '3q_128_cls_moe_128h_20250902_180934'"
good and terrible at the same time. 非常不稳定，还是应该先考虑增大dmodel
5. 4 head only 
python train_single.py\
 --exp_name 3q_128_cls_moe_128h\
 --num_qubits 3\
 --d_model 128\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 4\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4
6. 要分析学的特别不好的、一般的、各自有什么区别；有没有可能是数据过拟合？ - 总体看下来还是dmodel变大最有用呢 
   - 📊 [3q_256_cls_moe_64h] Epoch 18|val   | Loss: 0.010000 | MAE: 0.080239 | R²: 0.4319                              bro we get this 
🔍 Evaluating: 100%|██████████| 2/2 [00:02<00:00,  1.21s/it]
📋 [3q_256_cls_moe_64h] Final Test Results:
   MSE: 0.007081
   MAE: 0.065648
   R²:  0.5954
   Spearman: 0.7818
🔍 Evaluating: 100%|██████████| 2/2 [00:02<00:00,  1.21s/it]
📋 [3q_256_cls_moe_64h] Final Test Results:
   MSE: 0.007081
   MAE: 0.065648
   R²:  0.5954
   Spearman: 0.7818

✅ Experiment 3q_256_cls_moe_64h completed successfully!
📁 All outputs saved to: experiments/3q_256_cls_moe_64h_20250902_181813
📄 Training log: experiments/3q_256_cls_moe_64h_20250902_181813/3q_256_cls_moe_64h.csv
📄 Final metrics: experiments/3q_256_cls_moe_64h_20250902_181813/3q_256_cls_moe_64h_final_metrics.json
🤖 Model weights: models/best_3q_256_cls_moe_64h_20250902_181813.pth
⚙️ Configuration: experiments/3q_256_cls_moe_64h_20250902_181813/3q_256_cls_moe_64h_config.json

🔍 To visualize this model, run:
python Visualization.py \
    --model_path 'models/best_3q_256_cls_moe_64h_20250902_181813.pth' \
    --data_folder 'Decoded_Tokens' \
    --labels_path 'Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy' \
    --config_from_json 'experiments/3q_256_cls_moe_64h_20250902_181813/3q_256_cls_moe_64h_config.json' \
    --output_prefix '3q_256_cls_moe_64h_20250902_181813'

6. lets do larger dmodel with 512, 4 head change
python train_single.py\
 --exp_name 3q_512_cls_moe_4h\
 --num_qubits 3\
 --d_model 512\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 4\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4

🧪 [3q_512_cls_moe_4h] Final Test Evaluation...
/home/gwl/3q/MixMLP/train_single.py:350: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path_manager.model_path)) # Load best model
🔍 Evaluating: 100%|██████████| 2/2 [00:02<00:00,  1.30s/it]
📋 [3q_512_cls_moe_4h] Final Test Results:
   MSE: 0.002756
   MAE: 0.040264
   R²:  0.8425
   Spearman: 0.9113

✅ Experiment 3q_512_cls_moe_4h completed successfully!
📁 All outputs saved to: experiments/3q_512_cls_moe_4h_20250902_182736
📄 Training log: experiments/3q_512_cls_moe_4h_20250902_182736/3q_512_cls_moe_4h.csv
📄 Final metrics: experiments/3q_512_cls_moe_4h_20250902_182736/3q_512_cls_moe_4h_final_metrics.json
🤖 Model weights: models/best_3q_512_cls_moe_4h_20250902_182736.pth
⚙️ Configuration: experiments/3q_512_cls_moe_4h_20250902_182736/3q_512_cls_moe_4h_config.json
7. 在一个有效的基础上增大nhead
python train_single.py\
 --exp_name 3q_512_cls_moe_4h\
 --num_qubits 3\
 --d_model 512\
 --pooling_type cls\
 --mlp_type mixture_of_experts\
 --nhead 64\
 --epochs 20\
 --data_folder Decoded_Tokens\
 --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
 --num_workers 4


🧪 [3q_512_cls_moe_4h] Final Test Evaluation...
/home/gwl/3q/MixMLP/train_single.py:350: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path_manager.model_path)) # Load best model
🔍 Evaluating: 100%|██████████| 2/2 [00:02<00:00,  1.40s/it]
📋 [3q_512_cls_moe_4h] Final Test Results:
   MSE: 0.002991
   MAE: 0.043006
   R²:  0.8291
   Spearman: 0.9058
还是不知道啊。XAI吧。
✅ Experiment 3q_512_cls_moe_4h completed successfully!
📁 All outputs saved to: experiments/3q_512_cls_moe_4h_20250902_183252
📄 Training log: experiments/3q_512_cls_moe_4h_20250902_183252/3q_512_cls_moe_4h.csv
📄 Final metrics: experiments/3q_512_cls_moe_4h_20250902_183252/3q_512_cls_moe_4h_final_metrics.json
🤖 Model weights: models/best_3q_512_cls_moe_4h_20250902_183252.pth
⚙️ Configuration: experiments/3q_512_cls_moe_4h_20250902_183252/3q_512_cls_moe_4h_config.json

🔍 To visualize this model, run:
python Visualization.py \
    --model_path 'models/best_3q_512_cls_moe_4h_20250902_183252.pth' \
    --data_folder 'Decoded_Tokens' \
    --labels_path 'Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy' \
    --config_from_json 'experiments/3q_512_cls_moe_4h_20250902_183252/3q_512_cls_moe_4h_config.json' \
    --output_prefix '3q_512_cls_moe_4h_20250902_183252'

python train_single.py \
  --exp_name <NAME> \
  --num_qubits 3 \
  --d_model <D_MODEL> \
  --nhead <N_HEAD> \
  --pooling_type cls \
  --mlp_type asymmetric_ensemble \
  --use_physics_mask False \
  --mask_threshold 1.0 \
  --use_cls_token True \
  --epochs 100 \
  --lr 1e-4 \
  --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --num_workers 4 \
  --device auto \
  --base_log_dir experiments \
  --base_model_dir models

  # 本地测试可用
  # --dummy_test --dummy_samples 32

# 1) moe, d_model=512, nhead=4
python train_single.py --exp_name 3q_d512_h4_moe --num_qubits 3 --d_model 512 --nhead 4 \
  --pooling_type cls --mlp_type mixture_of_experts --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 2) moe, d_model=512, nhead=8
python train_single.py --exp_name 3q_d512_h8_moe --num_qubits 3 --d_model 512 --nhead 8 \
  --pooling_type cls --mlp_type mixture_of_experts --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 3) moe, d_model=1024, nhead=4
python train_single.py --exp_name 3q_d1024_h4_moe --num_qubits 3 --d_model 1024 --nhead 4 \
  --pooling_type cls --mlp_type mixture_of_experts --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 4) moe, d_model=1024, nhead=8

python train_single.py --exp_name 3q_d1024_h8_moe --num_qubits 3 --d_model 1024 --nhead 8 \
  --pooling_type cls --mlp_type mixture_of_experts --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 5) moe, d_model=2048, nhead=4
python train_single.py --exp_name 3q_d2048_h4_moe --num_qubits 3 --d_model 2048 --nhead 4 \
  --pooling_type cls --mlp_type mixture_of_experts --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models
==============
# 6) moe, d_model=2048, nhead=8
python train_single.py --exp_name 3q_d2048_h8_moe --num_qubits 3 --d_model 2048 --nhead 8 \
  --pooling_type cls --mlp_type mixture_of_experts --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 7) asymmetric, d_model=512, nhead=4
python train_single.py --exp_name 3q_d512_h4_asym --num_qubits 3 --d_model 512 --nhead 4 \
  --pooling_type cls --mlp_type asymmetric_ensemble --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 8) asymmetric, d_model=512, nhead=8
python train_single.py --exp_name 3q_d512_h8_asym --num_qubits 3 --d_model 512 --nhead 8 \
  --pooling_type cls --mlp_type asymmetric_ensemble --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 9) asymmetric, d_model=1024, nhead=4
python train_single.py --exp_name 3q_d1024_h4_asym --num_qubits 3 --d_model 1024 --nhead 4 \
  --pooling_type cls --mlp_type asymmetric_ensemble --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 10) asymmetric, d_model=1024, nhead=8
python train_single.py --exp_name 3q_d1024_h8_asym --num_qubits 3 --d_model 1024 --nhead 8 \
  --pooling_type cls --mlp_type asymmetric_ensemble --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 11) asymmetric, d_model=2048, nhead=4
python train_single.py --exp_name 3q_d2048_h4_asym --num_qubits 3 --d_model 2048 --nhead 4 \
  --pooling_type cls --mlp_type asymmetric_ensemble --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models

# 12) asymmetric, d_model=2048, nhead=8
python train_single.py --exp_name 3q_d2048_h8_asym --num_qubits 3 --d_model 2048 --nhead 8 \
  --pooling_type cls --mlp_type asymmetric_ensemble --use_physics_mask False --mask_threshold 1.0 \
  --use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
  --data_folder Decoded_Tokens \
  --labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
  --train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
  --device auto --base_log_dir experiments --base_model_dir models


512 is fine 4 head is also good lets try more layers 
Not changing the training dataset for the time being 
need to figure out new ways for XAI which explain more about magic.