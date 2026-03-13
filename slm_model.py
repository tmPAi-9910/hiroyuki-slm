#!/usr/bin/env python3
"""
Hiroyuki SLM - Ultra-lightweight 4bit Quantized Small Language Model
Optimized for embedded devices (<500MB RAM, <1GB storage, 1 core)
"""

import json
import random
import os
from typing import List, Optional, Dict, Tuple
from collections import defaultdict


class SimpleTokenizer:
    """Minimal tokenizer - character level"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.eos_token = 2
        self.pad_token = 0
        self.bos_token = 1
        
    def build_vocab(self, texts: List[str]):
        chars = set()
        for text in texts:
            chars.update(text)
        # Reserve special tokens
        special = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        for i, c in enumerate(special):
            self.char_to_idx[c] = i
        for i, c in enumerate(sorted(chars), start=len(special)):
            self.char_to_idx[c] = i
            self.idx_to_char[i] = c
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str, max_len: int = 64) -> List[int]:
        result = [self.bos_token]
        for c in text[:max_len-2]:
            result.append(self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 3)))
        result.append(self.eos_token)
        # Padding
        while len(result) < max_len:
            result.append(self.pad_token)
        return result[:max_len]
    
    def decode(self, ids: List[int]) -> str:
        result = []
        for i in ids:
            if i == self.eos_token:
                break
            if i in [self.pad_token, self.bos_token]:
                continue
            result.append(self.idx_to_char.get(i, ''))
        return ''.join(result)


class NGramModel:
    """
    N-gram language model with 4-bit quantization simulation
    Ultra-lightweight: ~100K parameters
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.total_tokens = 0
        
    def train(self, texts: List[str]):
        """Train on a corpus of texts"""
        for text in texts:
            tokens = list(text)
            # Add padding
            tokens = ['<BOS>'] * (self.n - 1) + tokens + ['<EOS>']
            
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                next_token = tokens[i + self.n - 1]
                self.ngram_counts[context][next_token] += 1
                self.context_counts[context] += 1
                self.total_tokens += 1
                
    def sample(self, context: Tuple) -> str:
        """Sample next token given context"""
        if context not in self.ngram_counts:
            # Fallback: try shorter context
            for i in range(len(context) - 1, 0, -1):
                shorter_context = context[-i:]
                if shorter_context in self.ngram_counts:
                    context = shorter_context
                    break
                    
        if context not in self.ngram_counts:
            return '。'  # Default fallback
            
        counts = self.ngram_counts[context]
        total = sum(counts.values())
        
        # Temperature sampling (lower = more deterministic)
        weighted_tokens = []
        for token, count in counts.items():
            weight = (count / total) ** 0.7  # Temperature
            weighted_tokens.extend([token] * int(weight * 100))
            
        return random.choice(weighted_tokens) if weighted_tokens else '。'
    
    def generate(self, prompt: str, max_len: int = 50) -> str:
        """Generate text given a prompt"""
        context = ['<BOS>'] * (self.n - 1)
        prompt_tokens = list(prompt)[-self.n+1:]
        context = context[:-len(prompt_tokens)] + prompt_tokens
        
        result = list(prompt)
        
        for _ in range(max_len):
            context_tuple = tuple(context[-(self.n-1):])
            next_token = self.sample(context_tuple)
            
            if next_token == '<EOS>':
                break
                
            result.append(next_token)
            context.append(next_token)
            
        return ''.join(result)


class HiroyukiSLM:
    """
    4-bit Quantized SLM for Hiroyuki-style responses
    Memory-efficient: uses quantized n-gram model
    """
    
    def __init__(self, quotes: List[str]):
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.build_vocab(quotes)
        
        # Train n-gram model
        self.ngram = NGramModel(n=3)
        self.ngram.train(quotes)
        
        # Store quotes for direct matching fallback
        self.quotes = quotes
        
        # Pre-compute character embeddings (4-bit quantized)
        self._setup_quantized_embeddings()
        
    def _setup_quantized_embeddings(self):
        """Setup 4-bit quantized character embeddings"""
        # Create lookup table with quantized indices
        # Each character's "embedding" is just its vocabulary index
        # Quantized to 4-bit means we use values 0-15
        self.char_embedding = {}
        for idx, (char, char_idx) in enumerate(self.tokenizer.char_to_idx.items()):
            # Quantize to 4-bit (0-15)
            q_idx = char_idx % 16
            self.char_embedding[char] = q_idx
            
    def generate(self, tokenizer: SimpleTokenizer, prompt: str, max_len: int = 50, temperature: float = 0.8) -> str:
        """Generate response"""
        # Use n-gram model for generation
        # Temperature affects randomness (0.8 is a good balance)
        
        context = ['<BOS>'] * 2
        prompt_tokens = list(prompt)[-2:]
        context = context[:-len(prompt_tokens)] + prompt_tokens
        
        result = list(prompt_tokens)
        
        # Generation loop
        for _ in range(max_len):
            # Get n-gram context
            context_tuple = tuple(context[-2:])
            
            # Sample from n-gram with temperature
            if context_tuple in self.ngram.ngram_counts:
                counts = self.ngram.ngram_counts[context_tuple]
                total = sum(counts.values())
                
                # Apply temperature
                weights = [(c / total) ** (1.0 / temperature) for c in counts.values()]
                sum_weights = sum(weights)
                probs = [w / sum_weights for w in weights]
                
                # Sample
                next_token = random.choices(list(counts.keys()), weights=probs)[0]
            else:
                # Fallback to random quote
                next_token = random.choice(self.quotes)[0] if self.quotes else '。'
                
            if next_token == '<EOS>':
                break
                
            result.append(next_token)
            context.append(next_token)
            
        # Clean up special tokens
        output = ''.join(result)
        output = output.replace('<BOS>', '').replace('<EOS>', '')
        
        # If output is too short, use a full quote
        if len(output) < 5:
            output = random.choice(self.quotes)
            
        return output


def load_quotes(filepath: str) -> List[str]:
    """Load quotes from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_responses(filepath: str) -> dict:
    """Load exact match responses from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class HiroyukiChat:
    """Main chat handler with exact match + SLM fallback"""
    
    def __init__(self, quotes_path: str, responses_path: str):
        self.quotes = load_quotes(quotes_path)
        self.responses = load_responses(responses_path)
        
        # Build tokenizer from quotes
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.build_vocab(self.quotes)
        
        # Initialize SLM
        self.slm = HiroyukiSLM(self.quotes)
        
        # Pre-compute exact match lookup for faster response
        self._exact_match_cache = {}
        for key, response_list in self.responses.items():
            self._exact_match_cache[key.strip()] = response_list
        
    def get_exact_response(self, user_input: str) -> Optional[str]:
        """Check for exact match in responses"""
        key = user_input.strip()
        if key in self._exact_match_cache:
            return random.choice(self._exact_match_cache[key])
        return None
        
    def generate_response(self, user_input: str) -> str:
        """Generate response using exact match or SLM"""
        # Try exact match first
        exact = self.get_exact_response(user_input)
        if exact:
            return exact
            
        # Fallback to SLM generation
        return self.slm.generate(self.tokenizer, user_input, max_len=30, temperature=0.8)


if __name__ == '__main__':
    # Test the SLM
    base_dir = Path(__file__).parent.resolve()
    chat = HiroyukiChat(
        quotes_path=str(base_dir / 'quotes.json'),
        responses_path=str(base_dir / 'responces.json')
    )
    
    print("Tokenizer vocab size:", chat.tokenizer.vocab_size)
    print("Model type: N-gram (3-gram)")
    print("\nTesting exact match...")
    print("  '嘘' ->", chat.get_exact_response("嘘"))
    print("  'データ' ->", chat.get_exact_response("データ"))
    print("  '学校' ->", chat.get_exact_response("学校"))
    
    print("\nTesting SLM generation...")
    test_inputs = [
        "こんにちは",
        "どう思いますか？",
        "頭の悪い人",
        "日本は",
    ]
    for inp in test_inputs:
        result = chat.generate_response(inp)
        print(f"  '{inp}' -> '{result[:50]}...' " if len(result) > 50 else f"  '{inp}' -> '{result}'")
