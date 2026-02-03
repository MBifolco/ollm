Summary for ChatGPT - A1 Early-Exit Simulation Results                                                                     
                                                                                                                             
  Key Findings                                                                                                               
  ┌────────────┬──────────┬──────────┬─────────────┬───────────────────┐                                                     
  │ Threshold  │ Baseline │  Token   │ Token Depth │ Potential Savings │                                                     
  ├────────────┼──────────┼──────────┼─────────────┼───────────────────┤                                                     
  │ AUC ≥ 0.90 │ never    │ layer 16 │ 67%         │ 33%               │                                                     
  ├────────────┼──────────┼──────────┼─────────────┼───────────────────┤                                                     
  │ AUC ≥ 0.95 │ never    │ layer 17 │ 71%         │ 29%               │                                                     
  ├────────────┼──────────┼──────────┼─────────────┼───────────────────┤                                                     
  │ AUC ≥ 0.98 │ never    │ never    │ -           │ -                 │                                                     
  └────────────┴──────────┴──────────┴─────────────┴───────────────────┘                                                     
  Token Model Trajectory                                                                                                     
  ┌───────┬───────┬───────┬────────────┐                                                                                     
  │ Layer │ Depth │  AUC  │ % of Final │                                                                                     
  ├───────┼───────┼───────┼────────────┤                                                                                     
  │ 15    │ 62%   │ 0.898 │ 91.8%      │                                                                                     
  ├───────┼───────┼───────┼────────────┤                                                                                     
  │ 16    │ 67%   │ 0.945 │ 96.6%      │                                                                                     
  ├───────┼───────┼───────┼────────────┤                                                                                     
  │ 17    │ 71%   │ 0.959 │ 98.0%      │                                                                                     
  ├───────┼───────┼───────┼────────────┤                                                                                     
  │ 21    │ 88%   │ 0.971 │ 99.2%      │                                                                                     
  ├───────┼───────┼───────┼────────────┤                                                                                     
  │ 24    │ 100%  │ 0.979 │ 100%       │                                                                                     
  └───────┴───────┴───────┴────────────┘                                                                                     
  Headline                                                                                                                   
                                                                                                                             
  Token model reaches 98% of final AUC at 71% depth (layer 17/24)                                                            
  → Potential to skip 29% of layers with early-exit                                                                          
  Baseline NEVER reaches equivalent separability at any layer                                                                
                                                                                                                             
  Framing                                                                                                                    
                                                                                                                             
  This is a simulation showing the ceiling for potential savings. The token model's decision crystallizes into a readable    
  form early enough that, with an early-exit mechanism:                                                                      
  - We could stop at layer 17 and still achieve 0.959 AUC                                                                    
  - The baseline cannot match this performance at any depth                                                                  
                                                                                                                             
  Actual FLOPs reduction requires implementing early-exit (future work), but this quantifies the opportunity.    