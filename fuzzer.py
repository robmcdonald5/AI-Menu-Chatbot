from typing import Dict, List, Tuple, Set
from Levenshtein import distance
import logging

# Create logger for fuzzer module
logger = logging.getLogger(__name__)

class MenuFuzzer:
    def __init__(self, menu_items: List[str], meats: List[str], rice: List[str], 
                 beans: List[str], toppings: List[str], debug: bool = False):
        self.debug = debug
        self.vocabulary = self._build_vocabulary(menu_items, meats, rice, beans, toppings)
        
        if self.debug:
            logger.debug(f"Initialized MenuFuzzer with {sum(len(words) for words in self.vocabulary.values())} total terms")

    def _build_vocabulary(self, menu_items: List[str], meats: List[str], 
                         rice: List[str], beans: List[str], toppings: List[str]) -> Dict[str, Set[str]]:
        vocabulary = {
            'menu_items': set(menu_items),
            'meats': set(meats),
            'rice': set(rice),
            'beans': set(beans),
            'toppings': set(toppings)
        }
        
        # Add common variations (e.g., singular/plural forms)
        for category, words in vocabulary.items():
            expanded_words = set()
            for word in words:
                expanded_words.add(word)
                # Add simple plural form if it doesn't end in 's'
                if not word.endswith('s'):
                    expanded_words.add(f"{word}s")
            vocabulary[category] = expanded_words
        
        return vocabulary

    def _should_attempt_correction(self, word: str, candidate: str) -> bool:
        if len(word) < 3:  # Don't correct very short words
            return False
            
        # First and last letters should match to avoid changing word meaning
        return (word[0].lower() == candidate[0].lower() and 
                word[-1].lower() == candidate[-1].lower())

    def _find_closest_match(self, word: str, max_distance: int = 2) -> Tuple[str, str, int]:
        if len(word) < 3:  # Don't correct very short words
            return word, '', -1
            
        min_distance = max_distance + 1
        best_match = word
        best_category = ''
        
        # Check each category's vocabulary
        for category, valid_words in self.vocabulary.items():
            for valid_word in valid_words:
                if not self._should_attempt_correction(word, valid_word):
                    continue
                    
                current_distance = distance(word.lower(), valid_word.lower())
                
                if current_distance < min_distance and current_distance <= max_distance:
                    min_distance = current_distance
                    best_match = valid_word
                    best_category = category
        
        if min_distance <= max_distance:
            return best_match, best_category, min_distance
        return word, '', -1

    def correct_text(self, text: str) -> Tuple[str, List[Tuple[str, str, str]]]:
        words = text.split()
        corrections = []
        corrected_words = []
        
        for word in words:
            # Skip words that are already in vocabulary
            if any(word.lower() in word_set for word_set in self.vocabulary.values()):
                corrected_words.append(word)
                continue
                
            corrected_word, category, dist = self._find_closest_match(word)
            
            if dist != -1:  # A correction was made
                corrections.append((word, corrected_word, category))
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        if self.debug and corrections:
            logger.debug("Fuzzy corrections made:")
            for original, corrected, category in corrections:
                logger.debug(f"  {original} -> {corrected} ({category})")
        
        return corrected_text, corrections