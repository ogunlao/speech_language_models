import torch
import heapq

class BeamSearchDecoder:
    def __init__(self, language_model,
                 space_index: int, eos_index :int,
                 acoustic_weight: float = 1, lm_weight: float = 1,
                 word_len_weight: float = 1, space_weight: float = 1,
                 beam_size: int = 1,):
        self.lm = language_model
        self.acoustic_weight = acoustic_weight
        self.word_len_weight = word_len_weight
        self.space_weight = space_weight
        self.lm_weight = lm_weight
        self.beam_size = beam_size
        self.space_index = space_index
    
    def compute_lm_prob(self, prev_tokens, curr_token):
        """Computes the conditional probability of the current token given previous tokens.

        Args:
            prev_tokens (list): List of previous tokens
            curr_token (int): current token to evaluate

        Returns:
            float: Conditional probability of the current token given previous tokens.
        """
        if self.lm_weight > 0.0 and self.lm:
            # TODO: Implement compute_prob method for language model
            return self.lm.compute_prob(prev_tokens, curr_token)
        return 0.0
    
    def __call__(self, acoustic_prob: torch.tensor, num_candidates: int | None = None,):
        """Performs beam search decoding.

        Args:
            acoustic_prob (torch.tensor): Log probability matrix of the acoustic model
            num_candidates (int, optional): Number of best candidates to return. Should not be greater than \
                beam size. Defaults to 1.

        Returns:
            list: list of best decoded paths
        """
        # vocab_len x T
        vocab_len, T = acoustic_prob.size()
        min_heap = []
        for t in range(T):
            acoustic_log_prob_t = acoustic_prob[:, t]
            
            # for first timestep, just add highest probs until beam size
            if t == 0 or not min_heap:
                total_space = 0.0
                for r in range(vocab_len):
                    heapq.heapify(min_heap)
                    heap_item = [acoustic_log_prob_t[r], [r]]
                    
                    if len(min_heap) == self.beam_size:
                        # check if current probability is better
                        min_log_prob, *tokens =  min_heap[0]
                        if min_log_prob < heap_item[0]:
                            heapq.heapreplace(min_heap, heap_item)
                    else:
                        heapq.heappush(min_heap, heap_item)
                continue
            
            min_heap2 = []
            for heap_item in min_heap: 
                score, tokens = heap_item
                for r in range(vocab_len):
                    # get lm probability
                    lang_model_prob = self.compute_lm_prob(tokens, r)
                        
                    # update probability
                    new_score_r = score + acoustic_log_prob_t[r] \
                        + self.lm_weight * lang_model_prob \
                        + self.word_len_weight
                    new_score_r -= self.space_weight if r == self.space_index else 0.0
                    
                    new_heap_item = [new_score_r, tokens + [r]]
                    if len(min_heap2) == self.beam_size:
                        if new_score_r > min_heap2[0][0]:
                            heapq.heapreplace(min_heap2, new_heap_item)
                        continue
                    heapq.heappush(min_heap2, new_heap_item)

            min_heap, min_heap2 = min_heap2, []

        if num_candidates:
            return heapq.nlargest(num_candidates, min_heap)
        
        return min_heap


if __name__ == "__main__":
    
    x = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]
    
    x = torch.tensor(x).transpose(1, 0)
    x = torch.log(torch.clamp(x, min=1e-10))
    
    bs = BeamSearchDecoder(language_model=None,
                           acoustic_weight=1.0,
                           lm_weight=0.0,
                           word_len_weight=0.0,
                           space_weight=0.0,
                           beam_size=3, # beam size of 1 corresponds to greedy search
                           space_index=-1,
                           eos_index=0,
                           )
    
    ans = bs(x, num_candidates=None)
    print(ans)
                        
                    
            
            
            
            
            
        
        
        
    