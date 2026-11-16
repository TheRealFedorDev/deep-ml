import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    
    total = len(labels)
    counts = Counter(labels)
    entropy = 0.0
    
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    if not examples:
        return 0.0
    
    total_entropy = calculate_entropy([ex[target_attr] for ex in examples])
    total_len = len(examples)
    
    # Групп приимеры
    subsets = {}
    for ex in examples:
        val = ex[attr]
        subsets.setdefault(val, []).append(ex)
        
    weighted_entropy = 0.0
    for subset in subsets.values():
        weight = len(subset) / total_len
        subset_labels = [ex[target_attr] for ex in subset]
        weighted_entropy += weight * calculate_entropy(subset_labels)
        
    return total_entropy - weighted_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    # Сортировка по убыванию частоты -x[1], затем по алфавиту x[0]
    return min(counts.items(), key=lambda x: (-x[1], x[0]))[0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [ex[target_attr] for ex in examples]
    
    # БС1: все примеры относятся к одному классу
    if len(set(labels)) == 1:
        return labels[0]
    
    # БС2: не осталось атрибутов для разбиения
    if not attributes:
        return majority_class(examples, target_attr)
    
    # max IG 
    best_attr = max(attributes, key=lambda a: calculate_information_gain(examples, a, target_attr))
    
    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]
    
    # unq best att
    attr_values = sorted(set(ex[best_attr] for ex in examples))
    
    for val in attr_values:
        subset = [ex for ex in examples if ex[best_attr] == val]
        if not subset:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            tree[best_attr][val] = learn_decision_tree(subset, remaining_attrs, target_attr)
            
    return tree
