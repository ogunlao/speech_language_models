import torch
import heapq

class BeamSearchDecoder:
    def __init__(self, language_model,
                 acoustic_weight: float = 1, lm_weight: float = 1,
                 word_len_weight: float = 1, space_weight: float = 1,
                 beam_size: int = 1, space_index: int = -1,):
        self.lm = language_model
        self.acoustic_weight = acoustic_weight
        self.word_len_weight = word_len_weight
        self.space_weight = space_weight
        self.lm_weight = lm_weight
        self.beam_size = beam_size
        self.space_index = space_index
    
    def compute_lm_prob(self, prev_tokens, curr_token):
        if self.lm_weight > 0.0 and self.lm:
            return self.lm.compute_prob(prev_tokens, curr_token)
        return 0.0
    
    def __call__(self, acoustic_prob: torch.tensor, max_decode=1,):
        """Performs beam search decoding.

        Args:
            acoustic_prob (torch.tensor): Log probability matrix of the acoustic model
            max_decode (int, optional): Number of best paths to return. Defaults to 1.

        Returns:
            _type_: _description_
        """
        # vocab_len x T
        vocab_len, T = acoustic_prob.size()
        min_heap = []
        for t in range(T):
            acoustic_log_prob_t = acoustic_prob[:, t]
            
            # for first timestep, just add probs until beam size
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
            for r in range(vocab_len):
                for heap_item in min_heap: 
                    score, tokens = heap_item
                    
                    # get lm probability
                    lang_model_prob = self.compute_lm_prob(tokens, r)
                        
                    # update probability
                    score = score + acoustic_log_prob_t[r] \
                        + self.lm_weight * lang_model_prob \
                        + self.word_len_weight \
                        + self.space_weight if r == self.space_index else 0.0
                    
                    new_heap_item = [score, tokens + [r]]
                    if len(min_heap2) == self.beam_size:
                        if min_heap2[0][0] < score:
                            heapq.heapreplace(min_heap2, new_heap_item)
                        continue
                    heapq.heappush(min_heap2, new_heap_item)
                    
            min_heap, min_heap2 = min_heap2, []
       
        return heapq.nlargest(max_decode, min_heap)


if __name__ == "__main__":
    x = torch.randint(high=10, size=(20, 10)).to(torch.float)
    x = torch.log(x)
    
    bs = BeamSearchDecoder(language_model=None,
                           acoustic_weight=0.0,
                           lm_weight=0.0,
                           word_len_weight=0.0,
                           space_weight=0.0,
                           beam_size=10,
                           space_index=-1,
                           )
    
    ans = bs(x, max_decode=1)
    print(ans)
                        
                    
            
            
            
            
            
        
        
        
    