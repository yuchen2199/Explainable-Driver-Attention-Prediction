import pdb
import copy
from collections import defaultdict
import numpy as np
import pdb
import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from scipy.stats.mstats import gmean

def compute_penalty_by_length(candidate_len, reference_len, alpha=1.0):
    """
    :param candidate_len: candidate length
    :param reference_len: reference length
    :param alpha: parameter to adjust the penalty
    :return: penalty score by difference in length
    """
    delta = abs(reference_len - candidate_len)

    return np.e ** (-(delta ** 2) / (float(alpha) * float(reference_len) ** 2))


def compute_penalty_by_repetition(candidate_sent, reference_sent, penalty_func=lambda x: np.exp(-x)):
    """
    :param candidate_sent: candidate sentence
    :param reference_sent: reference sentence
    :param penalty_func: penalty function
    :return: penalty score by repetition
    """
    # tokenize only words
    tokenizer = RegexpTokenizer(r'\w+')
    tokens_candidate = tokenizer.tokenize(candidate_sent)
    tokens_reference = tokenizer.tokenize(reference_sent)

    word_freq_candidate = Counter(tokens_candidate)
    word_freq_reference = Counter(tokens_reference)

    scores = []

    for word, freq in word_freq_candidate.items():
        # words in the reference and in the hypothesis
        if word_freq_reference.get(word, None) is not None:
            diff = abs(word_freq_reference[word] - freq)
            scores.append(penalty_func(diff))
        else:
            # words in the hypothesis but not in the reference
            scores.append(penalty_func(freq - 1))

    return gmean(np.array(scores))

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

class CiderRScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderRScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.test = []
        self.refs = []
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.refs.append(refs)
            self.test.append(test)
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref,
                sent_hyp, sent_ref, coefficient_rep=0.8, coefficient_len=0.2):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure cosine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                if length_ref == 0:
                    print('sent:', sent_ref)
                rep_penalty = compute_penalty_by_repetition(sent_hyp, sent_ref,
                                                                  penalty_func=lambda x: 1 / (1 + x))
                len_penalty = compute_penalty_by_length(length_hyp+1, length_ref+1)

                # vrama91: added a length based gaussian penalty
                val[n] *= (rep_penalty ** coefficient_rep) * (len_penalty ** coefficient_len)
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for i, (test, refs) in enumerate(zip(self.ctest, self.crefs)):
            if len(test) == 0:
                score_avg = 0
                # append score of an image to the score list
                scores.append(score_avg)
            else:
                # compute vector for test captions
                vec, norm, length = counts2vec(test)
                # compute vector for ref captions
                score = np.array([0.0 for _ in range(self.n)])
                for j, ref in enumerate(refs):
                    vec_ref, norm_ref, length_ref = counts2vec(ref)
                    score += sim(vec, vec_ref, norm, norm_ref, length, length_ref,
                                 sent_hyp=self.test[i], sent_ref=self.refs[i][j])
                # change by vrama91 - mean of ngram scores, instead of sum
                score_avg = np.mean(score)
                # divide by number of references
                score_avg /= len(refs)
                # multiply score by 10
                score_avg *= 10.0
                # append score of an image to the score list
                scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)

class CiderR:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.cider_scorer = CiderRScorer(n=self._n, sigma=self._sigma)

    def append(self, gts, res):
        self.cider_scorer += (res, gts)

    def compute_score(self):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        (score, scores) = self.cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr-R"