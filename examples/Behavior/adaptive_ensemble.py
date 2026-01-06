"""
adaptive_ensemble.py

"""

from collections import deque
import numpy as np


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
  
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


class ChunkedAdaptiveEnsembler:
    def __init__(self, pred_action_horizon, update_interval, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.update_interval = update_interval
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        
        self.action_history = [] 
        self.current_step = 0

    def reset(self):
        self.action_history = []
        self.current_step = 0
    
    def ensemble_action(self, new_action_chunk):
        self.action_history.append({
            'start_step': self.current_step,
            'actions': new_action_chunk
        })

    def step(self):
        self.action_history = [
            h for h in self.action_history 
            if h['start_step'] + len(h['actions']) > self.current_step
        ]

        relevant_preds = []
        
        for history_item in self.action_history:
            idx = self.current_step - history_item['start_step']
            
            if 0 <= idx < len(history_item['actions']):
                relevant_preds.append(history_item['actions'][idx])

        if not relevant_preds:
            raise ValueError(f"Step {self.current_step}: No actions available. Did you forget to initialize with an action chunk?")

        curr_act_preds = np.stack(relevant_preds)
        num_actions = curr_act_preds.shape[0]
        
        if num_actions == 1:
            final_action = curr_act_preds[0]
        else:
            ref = curr_act_preds[-1, :] 
            previous_pred = curr_act_preds
            
            dot_product = np.sum(previous_pred * ref, axis=1)
            norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
            norm_ref = np.linalg.norm(ref)
            cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

            weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
            weights = weights / weights.sum()

            final_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        self.current_step += 1
        
        return final_action
