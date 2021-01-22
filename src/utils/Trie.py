from collections import defaultdict
import time

class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nodes = defaultdict(TrieNode)  # Easy to insert new node.
        self.isword = False  # True for the end of the trie.
        self.sub = None


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, subs):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        words = word.split(' ')
        curr = self.root
        for w in words:
            curr = curr.nodes[w]
        curr.isword = True
        curr.sub = subs

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self.root
        for w in word:
            if w not in curr.nodes:
                return False
            curr = curr.nodes[w]
        return curr.isword

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curr = self.root
        for w in prefix:
            if w not in curr.nodes:
                return False
            curr = curr.nodes[w]
        return True

    def get_sub(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self.root
        for w in word:
            if w not in curr.nodes:
                return None
            curr = curr.nodes[w]
        return curr.sub

    def build_trie(self, tuples):
        for tuple in tuples:
            word = tuple[0]
            sub = tuple[1]
            self.insert(word, sub)

