import heapq
import re
import itertools  

def split_into_sentences(text):
    
    sentences = re.split(r'[.!?]\s*', text.strip())
    return [s.lower().strip() for s in sentences if s.strip()]

def compute_levenshtein(s1, s2):
    
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution
    return dp[len1][len2]

def align_sentences_astar(doc1, doc2):
    
    sentences1 = split_into_sentences(doc1)
    sentences2 = split_into_sentences(doc2)
    n1, n2 = len(sentences1), len(sentences2)

    priority_queue = []
    counter = itertools.count()  # Tie-breaker for heap
    start = (0, 0)
    heapq.heappush(priority_queue, (0, next(counter), start, []))  # (f_cost, count, state, path)
    visited = set()

    def heuristic(i, j):
        
        return abs((n1 - i) - (n2 - j))

    total_cost = 0
    while priority_queue:
        f_cost, _, (i, j), path = heapq.heappop(priority_queue)

        if (i, j) in visited:
            continue
        visited.add((i, j))

        if i == n1 and j == n2:
            for _, dist in path:
                total_cost += dist
            return path, total_cost

   
        if i < n1 and j < n2:
            edit_dist = compute_levenshtein(sentences1[i], sentences2[j])
            g_cost = f_cost - heuristic(i, j) + edit_dist
            f_new = g_cost + heuristic(i + 1, j + 1)
            heapq.heappush(priority_queue, (f_new, next(counter), (i + 1, j + 1), path + [((i, j), edit_dist)]))

    
        if i < n1:
            g_cost = f_cost - heuristic(i, j) + 1
            f_new = g_cost + heuristic(i + 1, j)
            heapq.heappush(priority_queue, (f_new, next(counter), (i + 1, j), path + [((i, None), 1)]))

        
        if j < n2:
            g_cost = f_cost - heuristic(i, j) + 1
            f_new = g_cost + heuristic(i, j + 1)
            heapq.heappush(priority_queue, (f_new, next(counter), (i, j + 1), path + [((None, j), 1)]))

    return None, float('inf')

def detect_plagiarism(alignment, doc1, doc2, threshold=2):
   
    sentences1 = split_into_sentences(doc1)
    sentences2 = split_into_sentences(doc2)
    plagiarised_pairs = []
    for ((i, j), dist) in alignment:
        if i is not None and j is not None and dist <= threshold:
            plagiarised_pairs.append((sentences1[i], sentences2[j], dist))
    return plagiarised_pairs

if __name__ == "__main__":
    try:
        docA = "Artificial intelligence is fascinating. It is a growing field. Many applications utilize AI."
        docB = "Artificial intelligence is fascinating. AI is rapidly advancing. Applications use AI in many ways."

        print("Running A* Sentence Alignment for Plagiarism Detection...")
        alignment, total_cost = align_sentences_astar(docA, docB)

        if alignment:
            plag = detect_plagiarism(alignment, docA, docB)
            print(f"\nPotential plagiarism found (Total Edit Distance: {total_cost}):")
            for idx, (s1, s2, dist) in enumerate(plag, 1):
                print(f"Pair {idx}:")
                print(f'  Doc1: "{s1}"')
                print(f'  Doc2: "{s2}"')
                print(f'  Edit Distance: {dist}\n')
            if not plag:
                print("No plagiarism detected (edit distance <= 2).")
        else:
            print("No alignment found.")
    except Exception as e:
        print(f"An error occurred: {e}")
