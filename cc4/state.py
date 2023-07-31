import os
import copy


g_unique_valids_list = []

class State:
    def __init__(self, s=None):
        if s is not None:
            self._array = s
            self._c = len(self._array[0])
            self._r = len(self._array)
            self._valids_locs_dict = {}
            for i in range(self._r):
                for j in range(self._c):
                    num = self._array[i][j]
                    if num > 0:
                        if self._valids_locs_dict.get(num) is None:
                            self._valids_locs_dict[num] = []
                        self._valids_locs_dict[num].append((i, j))

    def copy(self):
        result = State()
        result._array = copy.deepcopy(self._array)
        result._c = self._c
        result._r = self._r
        result._valids_locs_dict = copy.deepcopy(self._valids_locs_dict)
        return result

    def to_string(self):
        result = ""

        # Traverse the rows in the 2D array
        for row in self._array:
            # Join the elements of each row and add a newline character
            row_str = " ".join(str(element) for element in row)
            result += row_str + "\n"

        return result

    def __eq__(self, other):
        return self.to_string() == other.to_string()

    def __lt__(self, other):
        return self.to_string() < other.to_string()

    def can_it_go_direction(self, num, direction):
        for location in self._valids_locs_dict[num]:
            row = location[0]
            col = location[1]
            if direction == 'U':
                if row == 0 or (self._array[row - 1][col] != 0 and self._array[row - 1][col] != num):
                    return False
            if direction == 'D':
                if row == (self._r-1) or (self._array[row + 1][col] != 0 and self._array[row + 1][col] != num):
                    return False
            if direction == 'L':
                if col == 0 or (self._array[row][col - 1] != 0 and self._array[row][col-1] != num):
                    return False
            if direction == 'R':
                if col == (self._c-1) or (self._array[row][col + 1] != 0 and self._array[row][col + 1] != num):
                    return False
        return True

    def get_actions(self):
        actions = []
        for unique_valid in g_unique_valids_list:
            if self.can_it_go_direction(unique_valid, 'U'):
                actions.append((unique_valid, 'U'))
            if self.can_it_go_direction(unique_valid, 'D'):
                actions.append((unique_valid, 'D'))
            if self.can_it_go_direction(unique_valid, 'L'):
                actions.append((unique_valid, 'L'))
            if self.can_it_go_direction(unique_valid, 'R'):
                actions.append((unique_valid, 'R'))

        return actions

    def apply_action(self, a):

        valid_actions = self.get_actions()
        assert a in valid_actions

        new_state = self.copy()

        action_num = a[0]
        action_dir = a[1]

        new_locs = []
        for location in self._valids_locs_dict[action_num]:
            row = location[0]
            col = location[1]
            new_row = row
            new_col = col
            if action_dir == 'U':
                new_row = row - 1
            if action_dir == 'D':
                new_row = row + 1
            if action_dir == 'L':
                new_col = col - 1
            if action_dir == 'R':
                new_col = col + 1
            new_state._array[row][col] = 0
            new_locs.append((new_row, new_col))

        for location in new_locs:
            new_row = location[0]
            new_col = location[1]
            new_state._array[new_row][new_col] = action_num

        new_state._valids_locs_dict[action_num] = new_locs

        return new_state

    def get_manhattan_distance(self, other):
        total_distance = 0
        global g_unique_valids_list
        for unique_valid in g_unique_valids_list:
            self_location = min(self._valids_locs_dict[unique_valid], key=lambda x: (x[0], x[1]))
            other_location = min(other._valids_locs_dict[unique_valid], key=lambda x: (x[0], x[1]))
            if self_location == other_location:
                continue
            diff = abs(self_location[0] - other_location[0]) + abs(self_location[1] - other_location[1])
            total_distance += diff
        return total_distance

    def is_same(self, other):
        return self.get_manhattan_distance(other) == 0

    def get_num_of_rows(self):
        return self._r

    def get_num_of_cols(self):
        return self._c

    def get_value_at(self, row, col):
        return self._array[row][col]


def create_unique_valids_list(init_state):
    temp_list = []
    for i in range(init_state.get_num_of_rows()):
        for j in range(init_state.get_num_of_cols()):
            num = init_state.get_value_at(i, j)
            if num > 0:
                temp_list.append(num)
    global g_unique_valids_list
    g_unique_valids_list = copy.deepcopy(sorted(set(temp_list)))