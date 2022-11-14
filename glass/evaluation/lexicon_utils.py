import Levenshtein as lstn
from tqdm import tqdm

def find_match_word(rec_str, lexicon, pairs, scores_numpy, use_ed = True, weighted_ed = False, text_encoder=None):
    ''' From MTSv3 evaluation code - see 'prepare_results.py' '''
    if not use_ed:
        return rec_str
    # rec_str = rec_str.upper()
    dist_min = 100
    dist_min_pre = 100
    match_word = ''
    match_dist = 100
    if not weighted_ed:
        for word in lexicon:
            word = word.upper()
            rec_str = rec_str.upper()
            ed = lstn.distance(rec_str, word)
            length_dist = abs(len(word) - len(rec_str))
            # dist = ed + length_dist
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist
    else:
        small_lexicon_dict = dict()
        for word in lexicon:
            word = word.upper()
            ed = lstn.distance(rec_str.upper(), word)
            small_lexicon_dict[word] = ed
            dist = ed
            if dist < dist_min_pre:
                dist_min_pre = dist
        small_lexicon = []
        for word in small_lexicon_dict:
            if small_lexicon_dict[word] <= dist_min_pre + 2:
                small_lexicon.append(word)

        for word in small_lexicon:
            word = word.upper()
            ed = weighted_edit_distance(rec_str, word, scores_numpy, text_encoder)
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist


def get_lexicon(dataset='totaltext', lexicon_type=2):
    if lexicon_type == 0:
        return None, None
    if dataset == 'totaltext':
        # weak lexicon
        lexicon_path = '/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/totaltext/weak_voc_new.txt'
        lexicon_fid = open(lexicon_path, 'r')
        pair_list = open('/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/totaltext/weak_voc_pair_list.txt', 'r')
        pairs = dict()
        for line in pair_list.readlines():
            line = line.strip()
            word = line.split(' ')[0].upper()
            word_gt = line[len(word) + 1:]
            pairs[word] = word_gt
        lexicon_fid = open(lexicon_path, 'r')
        lexicon = []
        for line in lexicon_fid.readlines():
            line = line.strip()
            lexicon.append(line)
        return lexicon, pairs
    elif dataset =='icdar15':
        if lexicon_type == 1:
            # generic lexicon
            lexicon_path = '/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/ic15/GenericVocabulary_new.txt'
            lexicon_fid = open(lexicon_path, 'r')
            pair_list = open('/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/ic15/GenericVocabulary_pair_list.txt', 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(lexicon_path, 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
            return lexicon, pairs
        if lexicon_type == 2:
            # weak lexicon
            lexicon_path = '/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/ic15/ch4_test_vocabulary_new.txt'
            lexicon_fid = open(lexicon_path, 'r')
            pair_list = open('/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/ic15/ch4_test_vocabulary_pair_list.txt', 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line = line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word) + 1:]
                pairs[word] = word_gt
            lexicon_fid = open(lexicon_path, 'r')
            lexicon = []
            for line in lexicon_fid.readlines():
                line = line.strip()
                lexicon.append(line)
            return lexicon, pairs

        if lexicon_type == 3:
            lexicon_dict = {}
            pairs_dict = {}
            for i in tqdm(range(1, 501)):
                img = 'img_' + str(i) + '.jpg'
                gt_img = 'gt_img_' + str(i) + '.txt'
                # weak
                lexicon_path = '/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/ic15/new_strong_lexicon/new_voc_img_' + str(i) + '.txt'
                lexicon_fid = open(lexicon_path, 'r')
                pair_list = open('/hiero_efs/HieroUsers/ilavi/HieroDeploy/weakly_sup/WeaklySupervisedOCR/MaskTextSpotterV3/evaluation/lexicons/ic15/new_strong_lexicon/pair_voc_img_' + str(i) + '.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line = line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word) + 1:]
                    pairs[word] = word_gt
                lexicon_fid = open(lexicon_path, 'r')
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)
                lexicon_dict[i] = lexicon
                pairs_dict[i] = pairs
            return lexicon_dict, pairs_dict
    raise ValueError('No lexicon for dataset: {0}, and type: {1}'.format(dataset, str(lexicon_type)))


''' Weighted Edit Distance - from MTSv3 eval - see weighted_editdistance.py '''

def weighted_edit_distance(word1, word2, scores, text_encoder):
    m = len(word1)
    n = len(word2)
    dp = [[0 for __ in range(m + 1)] for __ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):  ## word2
        for j in range(1, m + 1): ## word1
            delect_cost = ed_delect_cost(j-1, i-1, word1, word2, scores, text_encoder)  ## delect a[i]
            insert_cost = ed_insert_cost(j-1, i-1, word1, word2, scores, text_encoder)  ## insert b[j]
            if word1[j - 1].upper() != word2[i - 1].upper():
                replace_cost = ed_replace_cost(j-1, i-1, word1, word2, scores, text_encoder) ## replace a[i] with b[j]
            else:
                replace_cost = 0
            dp[i][j] = min(dp[i-1][j] + insert_cost, dp[i][j-1] + delect_cost, dp[i-1][j-1] + replace_cost)

    return dp[n][m]


def ed_delect_cost(j, i, word1, word2, scores, text_encoder):
    ## delect a[i]
    c = text_encoder.char_encode(word1[j])
    return scores[j][c]


def ed_insert_cost(i, j, word1, word2, scores, text_encoder):
    ## insert b[j]
    if i < len(word1) - 1:
        c1 = text_encoder.char_encode(word1[i])
        c2 = text_encoder.char_encode(word1[i+1])
        return (scores[i][c1] + scores[i+1][c2])/2
    else:
        c1 = text_encoder.char_encode(word1[i])
        return scores[i][c1]


def ed_replace_cost(i, j, word1, word2, scores, text_encoder):
    ## replace a[i] with b[j]
    c1 = text_encoder.char_encode(word1[i])
    c2 = text_encoder.char_encode(word2[j])
    # c3 = text_encoder.char_encode(word2[j].lower()) # case sensitive
    # if word1 == "eeatpisaababarait".upper():
    #     print(scores[c2][i]/scores[c1][i])

    return max(1 - scores[i][c2]/scores[i][c1]*5, 0)# case sensitive