# Evaluation

## Trigram Language Model (Design Summary)

### How I stored the n-gram counts
I used a nested dictionary structure where each pair of words `(w1, w2)` points to a Counter that stores the possible next words.  
Example structure:
- tri[(w1, w2)][w3] = count  
This made it easy to update counts and also easy to sample the next word during generation.  
I also kept a unigram Counter to fall back on when a trigram context was not found.

### Text cleaning, padding, and unknown words
I cleaned the text using a small regex that extracts words and numbers.  
All words were converted to lowercase to keep the vocabulary simple.  
For padding, I added `<s>` and `<s>` at the start and added `</s>` at the end.  
I did not include an `<unk>` token because the dataset (Alice in Wonderland) is large enough, and keeping only known words makes the generation cleaner.

### Generate function and probabilistic sampling
The generate function starts with `<s> <s>` and predicts one word at a time.  
For each prediction, I used `random.choices` with trigram counts as weights.  
This automatically turns counts into probabilities and produces more natural sentences instead of repeating the most common word.  
If a trigram context was missing, the model fell back to unigram sampling so text generation always continued smoothly.

### Other design choices
- I kept the code simple, with short variable names and one-line comments.
- The model works directly on raw text files from the Project Gutenberg dataset.
- The logic is easy to follow: clean → tokenize → pad → count → sample.
- The design focuses on clarity and correctness instead of complexity, which fits the purpose of the assignment.


## Task 2 – Scaled Dot-Product Attention (Design Summary)

### How I implemented the attention function
I wrote the function `scaled_dot_product_attention(Q, K, V, mask=None)` using only NumPy.  
The goal was to follow the exact formula from the Transformer paper.  
The steps I used are:

1. Convert Q, K, and V into NumPy arrays.
2. Check that the shapes match (Q and K must have same depth, K and V must have same number of rows).
3. Compute the raw attention scores using `Q @ K.T`.
4. Divide the scores by `sqrt(dk)` to keep the values stable.
5. Apply a mask if given. Masked positions are replaced with a very large negative number so that softmax makes them zero.
6. Apply a stable softmax to turn the scores into probabilities.
7. Multiply the attention weights with V to get the final output.

### Handling the mask
The mask accepts both boolean values and 0/1 values.  
I also allowed the mask to broadcast automatically to the shape of the score matrix.  
If the mask shape does not match and cannot broadcast, the function raises a clear error.  
This keeps the behavior simple but still consistent with how masks work in Transformer models.

### Sampling and final output
Attention does not involve random sampling.  
Instead, the softmax converts the scaled scores into weights that sum to 1.  
These weights decide how much each value vector contributes to the final result.  
The function returns two things:
- the attention output (weighted sum of V)
- the attention weights (softmax results)

### Demonstration script
I added a small demo in `demo_attention.py`.  
It tests the function on small Q, K, and V matrices and prints:
- the attention weights  
- the final attended outputs  
- the effect of using a mask  

This confirms that the implementation works correctly and follows the math properly.

### Other design choices
- I kept variable names short and simple to match a student-level style.
- Comments are small and only explain the essential steps.
- The code avoids unnecessary complexity and focuses on showing understanding of the math.
- No external machine learning libraries were used; everything is done with NumPy as required.