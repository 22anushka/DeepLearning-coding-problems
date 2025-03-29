
# greedy search but at each level you pick "k" candidates
# all selected probabilities go to the model for autoregressive generation
import numpy as np

def beam_search(model, initial_tokens, beam_width=5, max_length=50, end_token_id=None):
    """
    Perform beam search decoding on a language model.
    
    Args:
        model: A function that takes a sequence of tokens and returns logits for the next token
        initial_tokens: Initial sequence of tokens to start generating from
        beam_width: Number of beams (candidate sequences) to maintain
        max_length: Maximum sequence length to generate
        end_token_id: Token ID that indicates the end of a sequence (optional)
        
    Returns:
        The most probable sequence generated
    """

    # Initialize with a single sequence
    sequences = [(initial_tokens, 0.0)]  # (sequence, score)
    
    # Generate until max_length is reached
    while len(sequences[0][0]) < max_length:
        # List to store all candidate expansions
        all_candidates = []
        
        # Flag to check if all sequences have ended
        all_ended = True
        
        # Expand each current sequence
        for seq, score in sequences:
            # Check if sequence has ended
            if end_token_id is not None and seq[-1] == end_token_id:
                # Keep this sequence as-is in candidates
                all_candidates.append((seq, score))
                continue
                
            all_ended = False
            
            # Get model predictions for next token
            logits = model(seq).logits # depending on the model mentioning attributes is Needed?
            
            # Convert to log probabilities
            log_probs = logits - np.log(np.sum(np.exp(logits)))
            
            # Get top beam_width tokens
            top_indices = np.argsort(log_probs)[-beam_width:]
            
            # Add each new token to the sequence
            for idx in top_indices:
                new_seq = seq + [idx]
                new_score = score + log_probs[idx]  # Add log probability (equals multiplying probabilities)
                all_candidates.append((new_seq, new_score))
        
        # If all sequences have ended, break
        if all_ended:
            break
            
        # Sort all candidates by score -> Important to refer
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top beam_width candidates
        sequences = all_candidates[:beam_width]
    return sequences[0][0] # first beam = best beam due to sorting
