from collections import defaultdict


class Eclat(object):
    def __init__(self, support):
        self._support = support

    def iter_count(self, iterable):
        inverted_table = self._build_inverted_table(iterable=iterable)
        while inverted_table:
            yield self._convert_to_count_map(inverted_table)
            inverted_table = self._generate_frequent_sets_from_lk(inverted_table)

    def count(self, iterable):
        frequent_sets = dict()
        for idx, freqs in enumerate(self.iter_count(iterable)):
            frequent_sets[idx+1] = freqs
        return freqs

    def _build_inverted_table(self, iterable):
        inverted_table = defaultdict(set)
        for user_idx, bucket in enumerate(iterable):
            for item in bucket:
                inverted_table[frozenset({item})].add(user_idx)
        return {
            key: val for key, val in inverted_table.items() if len(val) >= self._support
        }

    def _generate_frequent_sets_from_lk(self, frequent_sets_lk):
        next_lk = dict()
        not_frequent = set()
        for key1 in frequent_sets_lk:
            for key2 in frequent_sets_lk:
                set_union = key1 | key2
                if len(set_union) - len(key1) != 1 \
                        or set_union in next_lk \
                        or set_union in not_frequent:
                    continue
                elif not self._check_super_set(set_union, frequent_sets_lk):
                    not_frequent.add(set_union)
                else:
                    overlapped = frequent_sets_lk[key1] & frequent_sets_lk[key2]
                    if len(overlapped) >= self._support:
                        next_lk[set_union] = overlapped
                    else:
                        not_frequent.add(set_union)
        return next_lk

    @staticmethod
    def _check_super_set(super_set, frequent_sets_lk):
        for item in super_set:
            if (super_set - frozenset({item})) not in frequent_sets_lk:
                return False
        return True

    @staticmethod
    def _convert_to_count_map(inverted_table):
        return {
            key: len(val) for key, val in inverted_table.items()
        }


if __name__ == '__main__':
    eclat = Eclat(2)
    for s in eclat.iter_count([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]]):
        print(s)