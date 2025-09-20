import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class Meteor:

    def __init__(self, sep2w=False):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.meteor_cmd = ['/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm', '-a', 'data/paraphrase-en.gz']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                     cwd=os.path.dirname(os.path.abspath(__file__)), \
                     stdin=subprocess.PIPE, \
                     stdout=subprocess.PIPE, \
                     stderr=subprocess.PIPE,
                     env=self.env, universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.sep2w = sep2w
        if sep2w:
            self.meteor_p_wt = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)

            self.meteor_p_wy = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)
            self.eval_line_wt = 'EVAL'
            self.eval_line_wy = 'EVAL'
            self.scores_wt = []
            self.scores_wy = []
            self.count_wt = 0
            self.count_wy = 0
        self.lock = threading.Lock()
        self.scores = []
        self.eval_line = 'EVAL'
        self.count = 0

    def append(self, gts, res, mode=None):
        self.lock.acquire()
        if self.sep2w:
            if mode == 'what':
                stat = self._stat(res, gts, mode=mode)
                self.eval_line_wt += ' ||| {}'.format(stat)
                self.count_wt += 1
            elif mode == 'why':
                stat = self._stat(res, gts, mode=mode)
                self.eval_line_wy += ' ||| {}'.format(stat)
                self.count_wy += 1
        else:
            stat = self._stat(res, gts)
            self.eval_line += ' ||| {}'.format(stat)
            self.count += 1
        self.lock.release()

    def compute_score(self):
        self.lock.acquire()
        # Send to METEOR
        self.meteor_p.stdin.write(self.eval_line + '\n')

        if self.sep2w:
            self.meteor_p_wt.stdin.write(self.eval_line_wt + '\n')
            self.meteor_p_wt.stdin.write(self.eval_line_wy + '\n')

        # Collect segment scores
        for i in range(self.count):
            score = float(self.meteor_p.stdout.readline().strip())
            self.scores.append(score)

            if self.sep2w:
                score_wt = float(self.meteor_p_wt.stdout.readline().strip())
                self.scores_wt.append(score_wt)

                score_wy = float(self.meteor_p_wy.stdout.readline().strip())
                self.scores_wy.append(score_wy)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        if self.sep2w:
            final_score_wt = float(self.meteor_p_wt.stdout.readline().strip())
            final_score_wy = float(self.meteor_p_wy.stdout.readline().strip())
        self.lock.release()

        if self.sep2w:
            return final_score, self.scores, final_score_wt, self.scores_wt, final_score_wy, self.scores_wy

        return final_score, self.scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list, mode=None):
        hypothesis_str = hypothesis_str.replace('\n', ' ').replace('|||', '').replace('  ', ' ')
        reference_list[0] = reference_list[0].replace('\n', ' ').replace('|||', '').replace('  ', ' ')
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        if self.sep2w:
            if mode == 'what':
                self.meteor_p_wt.stdin.write(score_line + '\n')
                return self.meteor_p_wt.stdout.readline().strip()
            elif mode == 'why':
                self.meteor_p_wy.stdin.write(score_line + '\n')
                return self.meteor_p_wy.stdout.readline().strip()
            else:
                raise ValueError('mode must be "what" or "why"')
        else:
            self.meteor_p.stdin.write(score_line + '\n')
            return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        if self.sep2w:
            self.meteor_p_wt.stdin.close()
            self.meteor_p_wt.kill()
            self.meteor_p_wt.wait()

            self.meteor_p_wy.stdin.close()
            self.meteor_p_wy.kill()
            self.meteor_p_wy.wait()
        self.lock.release()